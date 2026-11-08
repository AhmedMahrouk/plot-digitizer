"""NE111 Term Project: Plot Digitizer.

A single-page Streamlit app that extracts x-y data from an image of a graph.

How it works
------------
Every clicked point is stored in *image pixel* coordinates. Data values are
worked out from those pixels whenever they are needed, using the calibration:

    P = O + a*u + b*v

where O is the origin click, u = (X click - O), v = (Y click - O), and (a, b)
is how far along each axis the point P lies. Because u and v are arbitrary
vectors rather than assumed to be horizontal and vertical, rotated scans work
with no extra steps. On a log axis the same interpolation is done on
log10(value) instead of value.

Storing pixels also means changing the calibration or the axis type instantly
updates every data value, and a zoomed view needs no special handling.

Run locally with:  streamlit run plot_digitizer.py
"""

import colorsys
import csv
import hashlib
import html
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from streamlit_image_coordinates import streamlit_image_coordinates

RENDER_WIDTH = 1100  # width in pixels of the image sent to the browser
OVERVIEW_WIDTH = 320  # width in pixels of the zoom navigator image
MAX_PORTRAIT_HEIGHT = 800  # keeps tall page scans on screen without scrolling
ZOOM_LEVELS = [1, 1.5, 2, 3, 4, 6, 8, 12, 16]
SIGNIFICANT_FIGURES = 7

CALIBRATION_NAMES = ["O", "X", "Y"]
CALIBRATION_COLOR = "#e6007e"
CALIBRATION_PROMPTS = [
    "**Calibration 1 of 3:** click the **origin**, where the x and y axes meet.",
    "**Calibration 2 of 3:** click a known tick **on the x-axis**, as far from "
    "the origin as you can for best accuracy.",
    "**Calibration 3 of 3:** click a known tick **on the y-axis**, as far from "
    "the origin as you can for best accuracy.",
]
PALETTE = [
    "#0072b2", "#e69f00", "#009e73", "#d55e00", "#7b3fa0",
    "#56b4e9", "#a6761d", "#1b9e77", "#666666", "#c9184a",
]

# Widgets whose values must survive if the image is briefly removed.
SETTING_DEFAULTS = {
    "x_label": "x",
    "y_label": "y",
    "x_scale": "Linear",
    "y_scale": "Linear",
    "x0": "0",
    "x1": "1",
    "y0": "0",
    "y1": "1",
    "zoom": 1,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def dataset_color(number):
    """Return a color for the nth dataset; every dataset gets its own color."""
    if number < len(PALETTE):
        return PALETTE[number]
    hue = (number * 0.618034) % 1.0  # golden ratio spacing keeps hues distinct
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.75)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def new_dataset():
    """Create an empty dataset. Numbers are never reused, nor are colors."""
    state = st.session_state
    number = state.dataset_counter
    state.dataset_counter += 1
    return {
        "id": number,
        "name": f"Dataset {number + 1}",
        "color": dataset_color(number),
        "points": [],  # list of (x_pixel, y_pixel) in the original image
    }


def reset_project():
    """Clear all clicks and settings, ready for a new image."""
    state = st.session_state
    state.calibration_points = []
    state.dataset_counter = 0
    first = new_dataset()
    state.datasets = [first]
    state.active_id = first["id"]
    state.table_version = 0
    state.view_cache = None
    state.overview_cache = None
    state.center = None
    for key, value in SETTING_DEFAULTS.items():
        state[key] = value


def init_state():
    state = st.session_state
    if "datasets" not in state:
        state.image = None
        state.image_hash = None
        state.last_click_time = None
        reset_project()
    # Streamlit forgets a widget's value on any run where it is not drawn.
    # Re-assigning keeps the settings if the user removes the uploaded image.
    for key in SETTING_DEFAULTS:
        state[key] = state[key]


def active_dataset():
    state = st.session_state
    return next(d for d in state.datasets if d["id"] == state.active_id)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image(raw_bytes):
    """Decode any supported upload into an 8-bit RGB image.

    Handles color, greyscale, palette (GIF), 16-bit/float TIFF and images with
    transparency. Multi-frame GIF and TIFF files use the first frame.
    """
    image = Image.open(io.BytesIO(raw_bytes))
    image.seek(0)
    image = ImageOps.exif_transpose(image)

    # High bit-depth greyscale would clip to white if converted directly.
    if image.mode.startswith("I") or image.mode == "F":
        values = np.asarray(image, dtype=float)
        low, high = np.nanmin(values), np.nanmax(values)
        span = (high - low) or 1.0
        image = Image.fromarray(((values - low) / span * 255).astype(np.uint8))

    # Transparent areas would otherwise turn black.
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image)

    return image.convert("RGB")


# ---------------------------------------------------------------------------
# Calibration maths
# ---------------------------------------------------------------------------
def to_axis(value, scale):
    return math.log10(value) if scale == "Log" else value


def from_axis(value, scale):
    return 10.0**value if scale == "Log" else value


def parse_number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def get_calibration():
    """Return (calibration, error). Both are None until three clicks exist."""
    state = st.session_state
    if len(state.calibration_points) < 3:
        return None, None

    x0, x1 = parse_number(state.x0), parse_number(state.x1)
    y0, y1 = parse_number(state.y0), parse_number(state.y1)
    if None in (x0, x1, y0, y1):
        return None, "The calibration values must all be numbers (e.g. 10, 0.5 or 1e-3)."
    if x0 == x1:
        return None, "The two x calibration values must be different."
    if y0 == y1:
        return None, "The two y calibration values must be different."
    if state.x_scale == "Log" and (x0 <= 0 or x1 <= 0):
        return None, "A log x-axis needs calibration values greater than 0."
    if state.y_scale == "Log" and (y0 <= 0 or y1 <= 0):
        return None, "A log y-axis needs calibration values greater than 0."

    origin, x_point, y_point = state.calibration_points
    ux, uy = x_point[0] - origin[0], x_point[1] - origin[1]
    vx, vy = y_point[0] - origin[0], y_point[1] - origin[1]
    determinant = ux * vy - uy * vx
    # |det| = |u||v|sin(angle), so this rejects axes that are nearly parallel.
    if abs(determinant) <= 0.05 * math.hypot(ux, uy) * math.hypot(vx, vy):
        return None, "The three calibration points are in a line. Please redo them."

    calibration = {
        "origin": origin,
        "u": (ux, uy),
        "v": (vx, vy),
        "det": determinant,
        "x_scale": state.x_scale,
        "y_scale": state.y_scale,
        "tx0": to_axis(x0, state.x_scale),
        "tx1": to_axis(x1, state.x_scale),
        "ty0": to_axis(y0, state.y_scale),
        "ty1": to_axis(y1, state.y_scale),
    }
    return calibration, None


def pixel_to_data(point, cal):
    """Convert an image pixel to (x, y) data values."""
    dx, dy = point[0] - cal["origin"][0], point[1] - cal["origin"][1]
    ux, uy = cal["u"]
    vx, vy = cal["v"]
    # Solve (dx, dy) = a*u + b*v using Cramer's rule.
    a = (dx * vy - dy * vx) / cal["det"]
    b = (ux * dy - uy * dx) / cal["det"]
    x = from_axis(cal["tx0"] + a * (cal["tx1"] - cal["tx0"]), cal["x_scale"])
    y = from_axis(cal["ty0"] + b * (cal["ty1"] - cal["ty0"]), cal["y_scale"])
    return x, y


def data_to_pixel(x, y, cal):
    """Convert (x, y) data values to an image pixel (used by table edits)."""
    a = (to_axis(x, cal["x_scale"]) - cal["tx0"]) / (cal["tx1"] - cal["tx0"])
    b = (to_axis(y, cal["y_scale"]) - cal["ty0"]) / (cal["ty1"] - cal["ty0"])
    ox, oy = cal["origin"]
    ux, uy = cal["u"]
    vx, vy = cal["v"]
    return ox + a * ux + b * vx, oy + a * uy + b * vy


def tilt_degrees(cal):
    """Rotation of the x-axis in the image (counter-clockwise is positive)."""
    return round(-math.degrees(math.atan2(cal["u"][1], cal["u"][0])), 1) + 0.0  # no "-0.0"


def tidy(value):
    """Round to a sensible number of significant figures."""
    return float(f"{value:.{SIGNIFICANT_FIGURES}g}")


# ---------------------------------------------------------------------------
# Zoomed view
# ---------------------------------------------------------------------------
def get_view_box():
    """Return the (left, top, right, bottom) region of the image on screen."""
    state = st.session_state
    width, height = state.image.size
    if state.center is None:
        state.center = (width / 2, height / 2)
    view_width, view_height = width / state.zoom, height / state.zoom
    left = min(max(state.center[0] - view_width / 2, 0), width - view_width)
    top = min(max(state.center[1] - view_height / 2, 0), height - view_height)
    return left, top, left + view_width, top + view_height


def pan(step_x, step_y):
    """Move the view by a fraction of its own width and height."""
    state = st.session_state
    left, top, right, bottom = get_view_box()
    view_width, view_height = right - left, bottom - top
    state.center = (
        left + view_width / 2 + step_x * view_width,
        top + view_height / 2 + step_y * view_height,
    )
    # Clamp, so panning past an edge cannot build up a hidden offset.
    left, top, right, bottom = get_view_box()
    state.center = ((left + right) / 2, (top + bottom) / 2)


# ---------------------------------------------------------------------------
# Click and edit callbacks (these run before the page is redrawn, so a new
# mark appears on the image straight away)
# ---------------------------------------------------------------------------
def on_image_click():
    state = st.session_state
    click = state.get("image_click")
    # A click can be delivered again on a later rerun; each one is counted once.
    if not click or click["unix_time"] == state.last_click_time:
        return
    state.last_click_time = click["unix_time"]
    left, top, right, bottom = state.view_box
    point = (
        left + (click["x"] + 0.5) * (right - left) / click["width"],
        top + (click["y"] + 0.5) * (bottom - top) / click["height"],
    )
    if len(state.calibration_points) < 3:
        state.calibration_points.append(point)
    else:
        active_dataset()["points"].append(point)


def on_overview_click():
    state = st.session_state
    click = state.get("overview_click")
    if not click:
        return
    width, height = state.image.size
    state.center = (click["x"] / click["width"] * width, click["y"] / click["height"] * height)


def on_select_dataset(dataset_id):
    st.session_state.active_id = dataset_id


def on_add_dataset():
    state = st.session_state
    dataset = new_dataset()
    state.datasets.append(dataset)
    state.active_id = dataset["id"]


def on_delete_dataset():
    state = st.session_state
    state.datasets = [d for d in state.datasets if d["id"] != state.active_id]
    if not state.datasets:
        state.datasets = [new_dataset()]
    state.active_id = state.datasets[-1]["id"]


def on_undo_point():
    points = active_dataset()["points"]
    if points:
        points.pop()


def on_rename(dataset, widget_key):
    new_name = st.session_state[widget_key].strip()
    if new_name:
        dataset["name"] = new_name


def on_table_edit(editor_key, calibration):
    """Apply table edits to the stored points, which moves their marks."""
    state = st.session_state
    datasets = {d["id"]: d for d in state.datasets}
    removals = []

    for row_text, changes in state[editor_key]["edited_rows"].items():
        row = int(row_text)
        # Group the edited cells in this row by dataset: {id: {"x": value}}.
        edited = {}
        for column, value in changes.items():
            dataset_id, axis = column.split("_")
            edited.setdefault(int(dataset_id), {})[axis] = value

        for dataset_id, values in edited.items():
            points = datasets[dataset_id]["points"]
            if row >= len(points):
                st.toast("To add a point, click it on the image.")
            elif any(value is None for value in values.values()):
                removals.append((dataset_id, row))
            else:
                x, y = pixel_to_data(points[row], calibration)
                x, y = values.get("x", x), values.get("y", y)
                if (calibration["x_scale"] == "Log" and x <= 0) or (
                    calibration["y_scale"] == "Log" and y <= 0
                ):
                    st.toast("Values on a log axis must be greater than 0.")
                else:
                    points[row] = data_to_pixel(x, y, calibration)

    # Remove from the bottom up so earlier row numbers stay valid.
    for dataset_id, row in sorted(removals, reverse=True):
        datasets[dataset_id]["points"].pop(row)

    # A fresh key discards the table's edit history, which is now applied.
    state.table_version += 1


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def get_label_font():
    return ImageFont.load_default(size=20)


def draw_calibration(draw, to_view):
    points = [to_view(p) for p in st.session_state.calibration_points]
    if len(points) >= 2:
        draw.line([points[0], points[1]], fill=CALIBRATION_COLOR, width=2)
    if len(points) == 3:
        draw.line([points[0], points[2]], fill=CALIBRATION_COLOR, width=2)
    for name, (x, y) in zip(CALIBRATION_NAMES, points):
        for width, color in ((6, "white"), (2, CALIBRATION_COLOR)):
            draw.line([(x - 12, y), (x + 12, y)], fill=color, width=width)
            draw.line([(x, y - 12), (x, y + 12)], fill=color, width=width)
        draw.text(
            (x + 12, y + 8), name, fill=CALIBRATION_COLOR,
            font=get_label_font(), stroke_width=3, stroke_fill="white",
        )


def draw_points(draw, to_view):
    radius = 5
    for dataset in st.session_state.datasets:
        for point in dataset["points"]:
            x, y = to_view(point)
            draw.ellipse(
                (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2),
                fill="white",
            )
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=dataset["color"])


def render_view(box):
    """Return the visible part of the image with all marks drawn on it."""
    state = st.session_state
    left, top, right, bottom = box
    scale = RENDER_WIDTH / (right - left)
    size = (RENDER_WIDTH, max(1, round((bottom - top) * scale)))

    # Resizing a big scan is the slow step, so reuse it until the view changes.
    if state.view_cache is None or state.view_cache[0] != box:
        base = state.image.resize(size, Image.Resampling.LANCZOS, box=box, reducing_gap=2.0)
        state.view_cache = (box, base)

    view = state.view_cache[1].copy()
    draw = ImageDraw.Draw(view)

    def to_view(point):
        return (point[0] - left) * scale, (point[1] - top) * scale

    draw_calibration(draw, to_view)
    draw_points(draw, to_view)
    return view


def render_overview(box):
    """Return a thumbnail of the whole image with the visible region outlined."""
    state = st.session_state
    width, height = state.image.size
    scale = OVERVIEW_WIDTH / width
    if state.overview_cache is None:
        size = (OVERVIEW_WIDTH, max(1, round(height * scale)))
        state.overview_cache = state.image.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0)
    overview = state.overview_cache.copy()
    ImageDraw.Draw(overview).rectangle([v * scale for v in box], outline="#e6007e", width=3)
    return overview


# ---------------------------------------------------------------------------
# Table and CSV
# ---------------------------------------------------------------------------
def dataset_values(dataset, calibration):
    return [tuple(tidy(v) for v in pixel_to_data(p, calibration)) for p in dataset["points"]]


def build_table(calibration):
    """Build the editable table: one x and one y column for each dataset."""
    state = st.session_state
    rows = max(len(d["points"]) for d in state.datasets)
    columns, config = {}, {}
    for dataset in state.datasets:
        values = dataset_values(dataset, calibration)
        values += [(None, None)] * (rows - len(values))
        x_column, y_column = f"{dataset['id']}_x", f"{dataset['id']}_y"
        columns[x_column] = [v[0] for v in values]
        columns[y_column] = [v[1] for v in values]
        config[x_column] = st.column_config.NumberColumn(f"{dataset['name']}: {state.x_label}")
        config[y_column] = st.column_config.NumberColumn(f"{dataset['name']}: {state.y_label}")
    return pd.DataFrame(columns, dtype=float), config


def build_csv(calibration):
    """Row 1 has the dataset names, row 2 the axis labels, then the values."""
    state = st.session_state
    table = [dataset_values(d, calibration) for d in state.datasets]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")

    header = []
    for dataset in state.datasets:
        header += [dataset["name"], ""]
    writer.writerow(header)
    writer.writerow([state.x_label, state.y_label] * len(state.datasets))

    for row in range(max(len(values) for values in table)):
        cells = []
        for values in table:
            if row < len(values):
                cells += [f"{v:.{SIGNIFICANT_FIGURES}g}" for v in values[row]]
            else:
                cells += ["", ""]
        writer.writerow(cells)

    text = buffer.getvalue()
    # Excel only reads accented or Greek labels correctly with a UTF-8 marker.
    return text.encode("utf-8" if text.isascii() else "utf-8-sig")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def sidebar_setup():
    state = st.session_state
    st.header("2. Axes")
    left, right = st.columns(2)
    left.text_input("X-axis name", key="x_label")
    right.text_input("Y-axis name", key="y_label")
    left.radio("X-axis scale", ["Linear", "Log"], key="x_scale", horizontal=True)
    right.radio("Y-axis scale", ["Linear", "Log"], key="y_scale", horizontal=True)

    st.header("3. Calibration values")
    st.caption(
        "The origin does not have to be (0, 0). Enter what the axes actually "
        "read at the points you click. Log axes need values above 0."
    )
    left, right = st.columns(2)
    left.text_input("x at origin (O)", key="x0")
    right.text_input("x at x-axis point (X)", key="x1")
    left.text_input("y at origin (O)", key="y0")
    right.text_input("y at y-axis point (Y)", key="y1")

    def undo_calibration():
        state.calibration_points.pop()

    def clear_calibration():
        state.calibration_points = []

    left, right = st.columns(2)
    has_points = bool(state.calibration_points)
    left.button("Undo click", on_click=undo_calibration, disabled=not has_points, width="stretch")
    right.button("Restart", on_click=clear_calibration, disabled=not has_points, width="stretch")


def sidebar_zoom():
    state = st.session_state
    st.header("Zoom")
    st.select_slider("Zoom level", ZOOM_LEVELS, key="zoom", format_func=lambda z: f"{z}x")
    if state.zoom == 1:
        return

    st.caption("Click the thumbnail to move the view.")
    streamlit_image_coordinates(
        render_overview(get_view_box()), key="overview_click", width="stretch",
        on_click=on_overview_click, cursor="pointer", image_format="JPEG", jpeg_quality=85,
    )
    for column, label, step in zip(
        st.columns(4),
        ("←", "↑", "↓", "→"),
        ((-0.5, 0), (0, -0.5), (0, 0.5), (0.5, 0)),
    ):
        column.button(label, on_click=pan, args=step, width="stretch", key=f"pan_{label}")


def show_status(calibration, error):
    state = st.session_state
    clicks = len(state.calibration_points)
    if clicks < 3:
        st.info(CALIBRATION_PROMPTS[clicks])
    elif error:
        st.error(error)
    else:
        st.success(
            f"Calibrated (image tilt {tilt_degrees(calibration):+.1f}°). "
            "Click the image to add points to the active dataset."
        )


def show_legend():
    state = st.session_state
    items = []
    for dataset in state.datasets:
        name = html.escape(dataset["name"])
        if dataset["id"] == state.active_id:
            name = f"<b>{name}</b>"
        items.append(
            f"<span style='white-space:nowrap;margin-right:1em'>"
            f"<span style='color:{dataset['color']}'>&#9679;</span> {name}</span>"
        )
    st.markdown(" ".join(items), unsafe_allow_html=True)


def show_image_panel():
    state = st.session_state
    show_legend()
    box = get_view_box()
    state.view_box = box  # the click callback needs the view the user saw

    width, height = state.image.size
    if height > width:
        container = st.container(width=max(320, round(MAX_PORTRAIT_HEIGHT * width / height)))
    else:
        container = st.container()
    with container:
        streamlit_image_coordinates(
            render_view(box), key="image_click", width="stretch", on_click=on_image_click,
            cursor="crosshair", image_format="JPEG", jpeg_quality=90,
        )


def show_dataset_panel():
    state = st.session_state
    st.subheader("4. Datasets")
    st.caption("New clicks go to the highlighted dataset. Click another to switch.")
    for item in state.datasets:
        count = len(item["points"])
        st.button(
            f"{item['name']} ({count} point{'' if count == 1 else 's'})",
            key=f"select_dataset_{item['id']}",
            type="primary" if item["id"] == state.active_id else "secondary",
            on_click=on_select_dataset,
            args=(item["id"],),
            width="stretch",
        )

    dataset = active_dataset()
    name_key = f"dataset_name_{dataset['id']}"
    st.text_input(
        "Name of active dataset", value=dataset["name"], key=name_key,
        on_change=on_rename, args=(dataset, name_key),
    )

    add_col, undo_col, delete_col = st.columns(3)
    add_col.button("New dataset", on_click=on_add_dataset, width="stretch")
    undo_col.button(
        "Undo point", on_click=on_undo_point, disabled=not dataset["points"], width="stretch"
    )
    with delete_col.popover("Delete", width="stretch"):
        st.write("Delete this dataset and all of its points?")
        st.button("Yes, delete it", on_click=on_delete_dataset, type="primary")


def show_data_panel(calibration):
    state = st.session_state
    st.subheader("5. Data")
    if calibration is None:
        st.caption("The table appears here once calibration is complete.")
        return

    if any(d["points"] for d in state.datasets):
        st.caption("Edit a cell to move its mark. Clear a cell to delete that point.")
        table, config = build_table(calibration)
        editor_key = f"table_{state.table_version}"
        st.data_editor(
            table, key=editor_key, column_config=config, hide_index=True, width="stretch",
            on_change=on_table_edit, args=(editor_key, calibration),
        )
    else:
        st.caption("No points yet. Click on the image to add some.")

    st.download_button(
        "Download CSV", build_csv(calibration), file_name="digitized_data.csv",
        mime="text/csv", type="primary", on_click="ignore", width="stretch",
    )


def main():
    st.set_page_config(page_title="Plot Digitizer", layout="wide")
    # Trim Streamlit's wide page margins so the graph gets more room.
    st.markdown(
        "<style>[data-testid='stMainBlockContainer']"
        "{padding:2.5rem 1.5rem 2rem 1.5rem}</style>",
        unsafe_allow_html=True,
    )
    init_state()
    state = st.session_state

    with st.sidebar:
        st.header("1. Image")
        uploaded = st.file_uploader(
            "Upload a graph image", type=["png", "jpg", "jpeg", "gif", "tif", "tiff"]
        )

    st.title("Plot Digitizer")
    if uploaded is None:
        st.write(
            "Extract x-y data from a picture of a graph. Upload an image in the "
            "sidebar, click three calibration points, then click your data."
        )
        return

    raw_bytes = uploaded.getvalue()
    image_hash = hashlib.sha256(raw_bytes).hexdigest()
    if image_hash != state.image_hash:
        try:
            image = load_image(raw_bytes)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.error("Sorry, that file could not be read as an image. Please try another one.")
            return
        reset_project()
        state.image = image
        state.image_hash = image_hash

    with st.sidebar:
        sidebar_setup()
        sidebar_zoom()

    calibration, error = get_calibration()
    image_column, data_column = st.columns([5, 3], gap="medium")
    with image_column:
        show_status(calibration, error)
        show_image_panel()
    with data_column:
        show_dataset_panel()
        show_data_panel(calibration)


main()
