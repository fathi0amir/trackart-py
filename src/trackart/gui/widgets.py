"""Small tkinter helpers used by all TrackArt tabs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

BG = "#f0f0f0"
FG = "#000000"


class Entry(tk.Entry):
    """An entry with a stored default value used by ``get_num``."""

    def __init__(self, parent, default="0", width=7, **kwargs):
        kwargs.setdefault("background", "white")
        super().__init__(parent, width=width, **kwargs)
        self.default = str(default)
        self.insert(0, str(default))

    def get_num(self):
        """Return the numeric value or the default when unparseable."""
        try:
            return float(self.get().strip())
        except ValueError:
            return float(self.default)

    def get_int(self):
        return int(round(self.get_num()))


def label(parent, text="", **kwargs):
    kwargs.setdefault("bg", BG)
    kwargs.setdefault("fg", FG)
    return tk.Label(parent, text=text, **kwargs)


def box(parent, title, **kwargs):
    """A titled group box (LabelFrame)."""
    kwargs.setdefault("text", title)
    kwargs.setdefault("bd", 1)
    kwargs.setdefault("relief", "groove")
    kwargs.setdefault("font", ("TkDefaultFont", 9, "bold"))
    return tk.LabelFrame(parent, **kwargs)


def lab_entry(parent, text, default="0", width=7, label_width=None):
    """A labelled entry in a horizontal row.  Returns (frame, entry)."""
    f = tk.Frame(parent, bg=BG)
    lab = label(f, text)
    if label_width:
        lab.configure(width=label_width, anchor="w")
    lab.pack(side="left", padx=(0, 4))
    e = Entry(f, default=default, width=width)
    e.pack(side="left")
    return f, e


def radio(parent, text, var, value, command=None, **kwargs):
    """A classic radio button (circle + label), like the MATLAB GUI."""
    kwargs.setdefault("bg", BG)
    kwargs.setdefault("fg", FG)
    kwargs.setdefault("anchor", "w")
    kwargs.setdefault("indicatoron", True)
    r = tk.Radiobutton(parent, text=text, variable=var, value=value,
                       command=command, **kwargs)
    return r


def check(parent, text, var, command=None, **kwargs):
    kwargs.setdefault("bg", BG)
    kwargs.setdefault("fg", FG)
    kwargs.setdefault("anchor", "w")
    c = tk.Checkbutton(parent, text=text, variable=var, command=command,
                       **kwargs)
    return c


def button(parent, text, command, width=None, **kwargs):
    kwargs.setdefault("bg", BG)
    kwargs.setdefault("relief", "raised")
    kwargs.setdefault("bd", 1)
    b = tk.Button(parent, text=text, command=command, width=width, **kwargs)
    return b


def make_grid(parent, columns, rows, sticky="nsew"):
    for c in range(columns):
        parent.columnconfigure(c, weight=1)
    for r in range(rows):
        parent.rowconfigure(r, weight=1)


def math_label(parent, tex, fontsize=14, dpi=220, bg="white", fg="black",
               max_width=None):
    """Render a mathtext equation to an image shown in a tk.Label.

    This mirrors the MATLAB GUI which displays the diffusion-model equations
    as rendered (image) text.  Falls back to a plain unicode label when the
    mathtext rendering fails.  Returns the label; the PhotoImage is kept alive
    on ``label.image``.
    """
    try:
        import io

        from matplotlib.mathtext import math_to_image
        from PIL import Image, ImageTk

        buf = io.BytesIO()
        math_to_image(tex, buf, dpi=dpi, format="png")
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        if max_width and img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, max(1, int(img.height * ratio))),
                             Image.LANCZOS)
        photo = ImageTk.PhotoImage(img, master=parent)
        lbl = tk.Label(parent, image=photo, bg=bg)
        lbl.image = photo  # keep a reference so the image is not garbage collected
        return lbl
    except Exception:
        # fall back to a plain text label
        return label(parent, tex, bg=bg, fg=fg)


def nf(x, nd=4):
    """Format a number for display (MATLAB num2str-ish)."""
    if x is None:
        return ""
    try:
        if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
            return f"{x:.3e}"
        return f"{x:.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


class ScrollFrame(tk.Frame):
    """A vertically scrollable frame for tall parameter panels.

    The content (packed into ``.inner``) scrolls when the window is shorter
    than the content, so no controls are ever clipped.  The canvas grows
    horizontally to fit the widest content row.
    """

    def __init__(self, parent, bg=BG):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0,
                                yscrollincrement=30)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", width=14)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.configure(command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind_all("<MouseWheel>", self._on_wheel)

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # grow the canvas so the content never clips horizontally
        req = self.inner.winfo_reqwidth()
        if req > self.canvas.winfo_width():
            self.canvas.configure(width=req)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._win, width=event.width)

    def _on_wheel(self, event):
        # scroll only when the pointer is over this scroll frame's content
        widget = self.winfo_containing(event.x_root, event.y_root)
        w = widget
        while w is not None:
            if w is self.inner or w is self.canvas:
                self.canvas.yview_scroll(int(-event.delta / 120), "units")
                return "break"
            try:
                w = w.master
            except Exception:
                break
        return None


class PlotPanel(tk.Frame):
    """A matplotlib figure embedded in the GUI (FigureCanvasTkAgg)."""

    def __init__(self, parent, nrows=1, ncols=1, figsize=(6, 5),
                 sharex=False, sharey=False, bg=BG):
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        super().__init__(parent, bg=bg)
        self.figure = Figure(figsize=figsize, dpi=100, facecolor="white")
        axes = self.figure.subplots(nrows, ncols, sharex=sharex,
                                    sharey=sharey)
        self.axes = list(np.asarray(axes).ravel())
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw(self):
        self.canvas.draw_idle()

    def clear(self):
        for ax in self.axes:
            ax.clear()

    def add_subplot(self, position):
        return self.figure.add_axes(position)
