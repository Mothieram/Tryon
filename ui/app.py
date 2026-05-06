"""
app.py - Premium Virtual Try-On Desktop GUI
Built with CustomTkinter for a modern, professional look.
Provides live camera preview, shirt selection, controls, and performance metrics.
"""

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import logging
from typing import Optional, List
from pathlib import Path

from engine.render_pipeline import RenderPipeline, PipelineStats
from engine.coreutils import setup_logger

logger = setup_logger("app")

# ─────────────────────────────────────────────
# Design Constants
# ─────────────────────────────────────────────

# Color palette: deep slate dark theme with accent
PALETTE = {
    "bg_primary":    "#0D1117",
    "bg_secondary":  "#161B22",
    "bg_card":       "#1C2128",
    "bg_hover":      "#22272E",
    "accent":        "#4F9DFF",
    "accent_hover":  "#6DB3FF",
    "accent_dim":    "#1A3558",
    "success":       "#3FB950",
    "warning":       "#D29922",
    "danger":        "#F85149",
    "text_primary":  "#E6EDF3",
    "text_secondary":"#8B949E",
    "text_dim":      "#484F58",
    "border":        "#30363D",
    "shadow":        "#0D1117",
}

# Configure theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class StatusBadge(ctk.CTkFrame):
    """Compact status indicator badge."""

    def __init__(self, master, label: str, **kwargs):
        super().__init__(master, fg_color=PALETTE["bg_card"],
                         corner_radius=8, **kwargs)

        self.label_text = ctk.CTkLabel(
            self, text=label,
            font=ctk.CTkFont(family="Helvetica", size=10),
            text_color=PALETTE["text_secondary"],
        )
        self.label_text.pack(side="left", padx=(8, 4), pady=4)

        self.value_label = ctk.CTkLabel(
            self, text="—",
            font=ctk.CTkFont(family="Helvetica", size=10, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        self.value_label.pack(side="left", padx=(0, 8), pady=4)

    def set_value(self, text: str, color: str = None):
        self.value_label.configure(text=text, text_color=color or PALETTE["text_primary"])


class ShirtThumbnailButton(ctk.CTkFrame):
    """Clickable shirt thumbnail card for catalog panel."""

    def __init__(self, master, name: str, idx: int, callback, thumbnail=None, **kwargs):
        super().__init__(
            master,
            fg_color=PALETTE["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=PALETTE["border"],
            **kwargs,
        )
        self.idx = idx
        self.callback = callback
        self._selected = False

        # Thumbnail image
        if thumbnail is not None:
            try:
                thumb_rgb = cv2.cvtColor(thumbnail[:, :, :3], cv2.COLOR_BGR2RGB)
                pil_thumb = Image.fromarray(thumb_rgb).resize((60, 75))
                ctk_thumb = ctk.CTkImage(light_image=pil_thumb, dark_image=pil_thumb, size=(60, 75))
                img_label = ctk.CTkLabel(self, image=ctk_thumb, text="")
                img_label.pack(pady=(8, 4))
                img_label.bind("<Button-1>", self._on_click)
            except Exception:
                pass

        # Name label
        short_name = name[:12] + "…" if len(name) > 12 else name
        self.name_label = ctk.CTkLabel(
            self, text=short_name,
            font=ctk.CTkFont(family="Helvetica", size=10),
            text_color=PALETTE["text_secondary"],
            wraplength=70,
        )
        self.name_label.pack(pady=(0, 8))
        self.name_label.bind("<Button-1>", self._on_click)
        self.bind("<Button-1>", self._on_click)

    def _on_click(self, event=None):
        if self.callback:
            self.callback(self.idx)

    def set_selected(self, selected: bool):
        self._selected = selected
        color = PALETTE["accent_dim"] if selected else PALETTE["bg_card"]
        border = PALETTE["accent"] if selected else PALETTE["border"]
        self.configure(fg_color=color, border_color=border)


class VirtualTryOnApp(ctk.CTk):
    """
    Premium Virtual Try-On Desktop Application.

    Layout:
    ┌──────────────────────────────────────────────────────┐
    │  Header: Logo + Title + Stats bar                    │
    ├─────────────────────────┬────────────────────────────┤
    │  Live Camera Preview    │  Control Panel             │
    │  (mirrored)             │  - Shirt Catalog           │
    │                         │  - Navigation              │
    │                         │  - Settings                │
    │                         │  - Performance             │
    └─────────────────────────┴────────────────────────────┘
    """

    PREVIEW_W = 720
    PREVIEW_H = 540
    PANEL_W = 320

    def __init__(self, pipeline: RenderPipeline):
        super().__init__()

        self.pipeline = pipeline
        self._cap: Optional[cv2.VideoCapture] = None
        self._camera_running = False
        self._camera_thread: Optional[threading.Thread] = None
        self._processed_frame: Optional[np.ndarray] = None
        self._last_stats = PipelineStats()
        self._shirt_cards: List[ShirtThumbnailButton] = []
        self._canvas_image_id = None
        self._preview_interval_ms = 33

        # App config
        self.title("🪡 Virtual Try-On Studio")
        self.geometry(f"{self.PREVIEW_W + self.PANEL_W + 60}x{self.PREVIEW_H + 140}")
        self.configure(fg_color=PALETTE["bg_primary"])
        self.resizable(True, True)
        self.minsize(900, 640)

        self._build_ui()
        self._refresh_shirt_catalog()
        self._update_loop()

    # ─────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all UI components."""
        self._build_header()
        self._build_main_area()
        self._build_status_bar()

    def _build_header(self):
        """Top header with branding and quick stats."""
        header = ctk.CTkFrame(
            self, fg_color=PALETTE["bg_secondary"],
            corner_radius=0, height=52,
        )
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        # Logo / Title
        logo_frame = ctk.CTkFrame(header, fg_color="transparent")
        logo_frame.pack(side="left", padx=20, pady=8)

        ctk.CTkLabel(
            logo_frame, text="✦ VIRTUAL TRY-ON",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="bold"),
            text_color=PALETTE["accent"],
        ).pack(side="left")

        ctk.CTkLabel(
            logo_frame, text="  STUDIO",
            font=ctk.CTkFont(family="Helvetica", size=16),
            text_color=PALETTE["text_secondary"],
        ).pack(side="left")

        # Quick stats badges
        badges_frame = ctk.CTkFrame(header, fg_color="transparent")
        badges_frame.pack(side="right", padx=20, pady=8)

        self._fps_badge = StatusBadge(badges_frame, "FPS")
        self._fps_badge.pack(side="right", padx=4)

        self._pose_badge = StatusBadge(badges_frame, "POSE")
        self._pose_badge.pack(side="right", padx=4)

        self._model_badge = StatusBadge(badges_frame, "AI")
        self._model_badge.pack(side="right", padx=4)

    def _build_main_area(self):
        """Main split layout: camera | controls."""
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=(12, 0))

        # ── Camera Preview (left) ─────────────────────────────
        cam_container = ctk.CTkFrame(
            main,
            fg_color=PALETTE["bg_secondary"],
            corner_radius=16,
            border_width=1,
            border_color=PALETTE["border"],
        )
        cam_container.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Canvas for video
        self._canvas = ctk.CTkCanvas(
            cam_container,
            bg=PALETTE["bg_primary"],
            highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True, padx=12, pady=12)

        # Camera placeholder text
        self._canvas.create_text(
            self.PREVIEW_W // 2, self.PREVIEW_H // 2,
            text="Click 'Start Camera' to begin",
            fill=PALETTE["text_dim"],
            font=("Helvetica", 14),
            tags="placeholder",
        )

        # ── Control Panel (right) ─────────────────────────────
        panel = ctk.CTkScrollableFrame(
            main,
            width=self.PANEL_W,
            fg_color=PALETTE["bg_secondary"],
            corner_radius=16,
            border_width=1,
            border_color=PALETTE["border"],
            scrollbar_fg_color=PALETTE["bg_card"],
        )
        panel.pack(side="right", fill="both", padx=(8, 0))

        self._build_camera_section(panel)
        self._build_shirt_section(panel)
        self._build_navigation_section(panel)
        self._build_settings_section(panel)
        self._build_performance_section(panel)

    def _section_header(self, parent, title: str, icon: str = ""):
        """Styled section header."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=12, pady=(16, 6))

        ctk.CTkLabel(
            frame,
            text=f"{icon}  {title}" if icon else title,
            font=ctk.CTkFont(family="Helvetica", size=11, weight="bold"),
            text_color=PALETTE["text_secondary"],
        ).pack(side="left")

        # Divider
        ctk.CTkFrame(
            frame, height=1,
            fg_color=PALETTE["border"],
        ).pack(side="right", fill="x", expand=True, padx=(8, 0))

    def _build_camera_section(self, parent):
        self._section_header(parent, "CAMERA", "📷")

        cam_frame = ctk.CTkFrame(parent, fg_color="transparent")
        cam_frame.pack(fill="x", padx=12, pady=4)

        self._start_btn = ctk.CTkButton(
            cam_frame,
            text="▶  Start Camera",
            command=self._toggle_camera,
            fg_color=PALETTE["success"],
            hover_color="#2EA043",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40,
            corner_radius=10,
        )
        self._start_btn.pack(fill="x", pady=(0, 6))

        self._screenshot_btn = ctk.CTkButton(
            cam_frame,
            text="📸  Screenshot",
            command=self._take_screenshot,
            fg_color=PALETTE["bg_card"],
            hover_color=PALETTE["bg_hover"],
            border_width=1,
            border_color=PALETTE["border"],
            font=ctk.CTkFont(size=12),
            height=36,
            corner_radius=10,
        )
        self._screenshot_btn.pack(fill="x", pady=(0, 6))

    def _build_shirt_section(self, parent):
        self._section_header(parent, "WARDROBE", "👕")

        self._catalog_frame = ctk.CTkScrollableFrame(
            parent,
            height=200,
            fg_color=PALETTE["bg_card"],
            corner_radius=10,
            scrollbar_fg_color=PALETTE["bg_secondary"],
        )
        self._catalog_frame.pack(fill="x", padx=12, pady=4)

        # Shirt grid placeholder
        ctk.CTkLabel(
            self._catalog_frame,
            text="Loading wardrobe…",
            text_color=PALETTE["text_dim"],
            font=ctk.CTkFont(size=11),
        ).pack(pady=20)

    def _build_navigation_section(self, parent):
        self._section_header(parent, "NAVIGATION", "◀▶")

        nav_frame = ctk.CTkFrame(parent, fg_color="transparent")
        nav_frame.pack(fill="x", padx=12, pady=4)

        prev_btn = ctk.CTkButton(
            nav_frame,
            text="◀  Prev",
            command=self._prev_shirt,
            fg_color=PALETTE["bg_card"],
            hover_color=PALETTE["bg_hover"],
            border_width=1,
            border_color=PALETTE["border"],
            font=ctk.CTkFont(size=12),
            height=38,
            corner_radius=10,
        )
        prev_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        next_btn = ctk.CTkButton(
            nav_frame,
            text="Next  ▶",
            command=self._next_shirt,
            fg_color=PALETTE["bg_card"],
            hover_color=PALETTE["bg_hover"],
            border_width=1,
            border_color=PALETTE["border"],
            font=ctk.CTkFont(size=12),
            height=38,
            corner_radius=10,
        )
        next_btn.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Active shirt indicator
        self._shirt_label = ctk.CTkLabel(
            parent,
            text="—",
            font=ctk.CTkFont(family="Helvetica", size=12, weight="bold"),
            text_color=PALETTE["accent"],
        )
        self._shirt_label.pack(pady=6)

    def _build_settings_section(self, parent):
        self._section_header(parent, "SETTINGS", "⚙")

        settings_frame = ctk.CTkFrame(parent, fg_color=PALETTE["bg_card"], corner_radius=10)
        settings_frame.pack(fill="x", padx=12, pady=4)

        # Opacity slider
        ctk.CTkLabel(
            settings_frame,
            text="Shirt Opacity",
            text_color=PALETTE["text_secondary"],
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=12, pady=(10, 2))

        self._opacity_slider = ctk.CTkSlider(
            settings_frame,
            from_=0.3,
            to=1.0,
            number_of_steps=14,
            command=self._on_opacity_change,
            progress_color=PALETTE["accent"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_hover"],
        )
        self._opacity_slider.set(0.95)
        self._opacity_slider.pack(fill="x", padx=12, pady=(0, 8))

        # Toggles
        toggle_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        toggle_frame.pack(fill="x", padx=12, pady=(0, 10))

        self._shadow_var = ctk.BooleanVar(value=True)
        shadow_cb = ctk.CTkCheckBox(
            toggle_frame,
            text="Dynamic Shadows",
            variable=self._shadow_var,
            command=lambda: self.pipeline.toggle_shadows(),
            font=ctk.CTkFont(size=11),
            text_color=PALETTE["text_secondary"],
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            checkmark_color="white",
        )
        shadow_cb.pack(anchor="w", pady=2)

        self._lighting_var = ctk.BooleanVar(value=True)
        lighting_cb = ctk.CTkCheckBox(
            toggle_frame,
            text="Lighting Adaptation",
            variable=self._lighting_var,
            command=lambda: self.pipeline.toggle_lighting(),
            font=ctk.CTkFont(size=11),
            text_color=PALETTE["text_secondary"],
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            checkmark_color="white",
        )
        lighting_cb.pack(anchor="w", pady=2)

        ctk.CTkFrame(settings_frame, fg_color=PALETTE["border"], height=1).pack(
            fill="x", padx=12, pady=(2, 6)
        )

        ctk.CTkLabel(
            settings_frame,
            text="Live AI Overlays",
            text_color=PALETTE["text_secondary"],
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=12, pady=(2, 2))

        overlay_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        overlay_frame.pack(fill="x", padx=12, pady=(0, 10))

        self._overlay_vars = {
            "holistic": ctk.BooleanVar(value=False),
            "parser": ctk.BooleanVar(value=False),
            "densepose": ctk.BooleanVar(value=False),
            "detectron": ctk.BooleanVar(value=False),
        }

        for text, key in [
            ("Holistic Pose", "holistic"),
            ("Parser", "parser"),
            ("DensePose", "densepose"),
            ("Detectron", "detectron"),
        ]:
            cb = ctk.CTkCheckBox(
                overlay_frame,
                text=text,
                variable=self._overlay_vars[key],
                command=lambda k=key: self._on_debug_toggle(k),
                font=ctk.CTkFont(size=11),
                text_color=PALETTE["text_secondary"],
                fg_color=PALETTE["accent"],
                hover_color=PALETTE["accent_hover"],
                checkmark_color="white",
            )
            cb.pack(anchor="w", pady=2)

    def _build_performance_section(self, parent):
        self._section_header(parent, "PERFORMANCE", "⚡")

        perf_frame = ctk.CTkFrame(parent, fg_color=PALETTE["bg_card"], corner_radius=10)
        perf_frame.pack(fill="x", padx=12, pady=(4, 16))

        metrics = [
            ("Pose Detect", "_pose_ms"),
            ("Cloth Warp", "_warp_ms"),
            ("Render", "_render_ms"),
            ("Total Frame", "_total_ms"),
        ]

        self._perf_labels: dict = {}
        for label, key in metrics:
            row = ctk.CTkFrame(perf_frame, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=3)

            ctk.CTkLabel(
                row, text=label,
                font=ctk.CTkFont(size=10),
                text_color=PALETTE["text_secondary"],
            ).pack(side="left")

            val = ctk.CTkLabel(
                row, text="—",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=PALETTE["text_primary"],
            )
            val.pack(side="right")
            self._perf_labels[key] = val

        # Add bottom padding
        ctk.CTkFrame(perf_frame, fg_color="transparent", height=4).pack()

    def _build_status_bar(self):
        """Bottom status bar."""
        status_bar = ctk.CTkFrame(
            self,
            fg_color=PALETTE["bg_secondary"],
            corner_radius=0,
            height=30,
        )
        status_bar.pack(fill="x", padx=0, pady=(8, 0), side="bottom")
        status_bar.pack_propagate(False)

        self._status_label = ctk.CTkLabel(
            status_bar,
            text="Ready • Stand in front of camera to try on shirts",
            font=ctk.CTkFont(size=10),
            text_color=PALETTE["text_dim"],
        )
        self._status_label.pack(side="left", padx=16)

        # Engine info
        self._engine_label = ctk.CTkLabel(
            status_bar,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=PALETTE["text_dim"],
        )
        self._engine_label.pack(side="right", padx=16)

    # ─────────────────────────────────────────────────────────
    # Camera Control
    # ─────────────────────────────────────────────────────────

    def _toggle_camera(self):
        if self._camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self, camera_idx: int = 0):
        """Start webcam capture in background thread."""
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0

        def open_camera(idx: int):
            if backend:
                return cv2.VideoCapture(idx, backend)
            return cv2.VideoCapture(idx)

        self._cap = open_camera(camera_idx)
        if not self._cap.isOpened():
            # Try alternative indices
            for idx in [1, 2]:
                self._cap = open_camera(idx)
                if self._cap.isOpened():
                    break

        if not self._cap.isOpened():
            logger.error("No camera found!")
            self._status_label.configure(
                text="❌ No camera detected. Connect a webcam and restart."
            )
            return

        # Set camera properties — single quality profile, no FPS toggle.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._camera_running = True
        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True
        )
        self._camera_thread.start()

        self._start_btn.configure(
            text="⏹  Stop Camera",
            fg_color=PALETTE["danger"],
            hover_color="#B91C1C",
        )
        self._status_label.configure(text="Camera running • Stand in front to try on shirts")

    def _stop_camera(self):
        self._camera_running = False
        if self._cap:
            self._cap.release()
            self._cap = None

        self._start_btn.configure(
            text="▶  Start Camera",
            fg_color=PALETTE["success"],
            hover_color="#2EA043",
        )
        self._processed_frame = None
        self._status_label.configure(text="Camera stopped")

    def _camera_loop(self):
        """Background thread: capture and process frames."""
        while self._camera_running and self._cap:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Mirror for natural selfie view
            frame = cv2.flip(frame, 1)

            # Process through pipeline
            try:
                processed, stats = self.pipeline.process_frame(frame)
                self._processed_frame = processed
                self._last_stats = stats
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                self._processed_frame = frame
            time.sleep(0.001)

    # ─────────────────────────────────────────────────────────
    # Shirt Controls
    # ─────────────────────────────────────────────────────────

    def _prev_shirt(self):
        self.pipeline.previous_shirt()
        self._update_shirt_selection()

    def _next_shirt(self):
        self.pipeline.next_shirt()
        self._update_shirt_selection()

    def _select_shirt(self, idx: int):
        self.pipeline.select_shirt(idx)
        self._update_shirt_selection()

    def _update_shirt_selection(self):
        """Update UI to reflect active shirt."""
        active_idx = self.pipeline._current_idx
        for card in self._shirt_cards:
            card.set_selected(card.idx == active_idx)

        garment = self.pipeline.current_garment
        if garment:
            name = garment.name.replace("_", " ")
            self._shirt_label.configure(text=f"👕 {name}")

    def _refresh_shirt_catalog(self):
        """Populate shirt catalog panel."""
        # Clear existing
        for widget in self._catalog_frame.winfo_children():
            widget.destroy()
        self._shirt_cards.clear()

        names = self.pipeline.get_shirt_names()
        thumbnails = self.pipeline.get_garment_thumbnails()

        if not names:
            ctk.CTkLabel(
                self._catalog_frame,
                text="No shirts loaded\nPlace PNG files in assets/shirts/",
                text_color=PALETTE["text_dim"],
                font=ctk.CTkFont(size=10),
                justify="center",
            ).pack(pady=20)
            return

        # Grid layout
        grid_frame = ctk.CTkFrame(self._catalog_frame, fg_color="transparent")
        grid_frame.pack(fill="x", padx=4, pady=4)

        col_count = 3
        for i, (name, thumb) in enumerate(zip(names, thumbnails)):
            row = i // col_count
            col = i % col_count

            card = ShirtThumbnailButton(
                grid_frame,
                name=name,
                idx=i,
                callback=self._select_shirt,
                thumbnail=thumb,
                width=82,
                height=110,
            )
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            self._shirt_cards.append(card)

        for c in range(col_count):
            grid_frame.columnconfigure(c, weight=1)

        self._update_shirt_selection()

    # ─────────────────────────────────────────────────────────
    # Settings Handlers
    # ─────────────────────────────────────────────────────────

    def _on_opacity_change(self, val: float):
        self.pipeline.set_opacity(val)

    def _on_debug_toggle(self, key: str):
        var = self._overlay_vars.get(key)
        if var is None:
            return
        self.pipeline.set_debug_overlay(key, bool(var.get()))

    def _take_screenshot(self):
        if self._processed_frame is not None:
            path = self.pipeline.take_screenshot(self._processed_frame)
            self._status_label.configure(text=f"Screenshot saved: {Path(path).name}")
        else:
            self._status_label.configure(text="No active frame to capture")

    # ─────────────────────────────────────────────────────────
    # UI Update Loop
    # ─────────────────────────────────────────────────────────

    def _update_loop(self):
        """Main UI refresh loop at ~30 FPS."""
        try:
            self._refresh_video()
            self._refresh_stats()
        except Exception as e:
            logger.debug(f"UI update error: {e}")

        self.after(self._preview_interval_ms, self._update_loop)

    def _refresh_video(self):
        """Update camera preview canvas."""
        frame = self._processed_frame
        if frame is None:
            return

        # Get canvas size
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        # Resize frame to fit canvas
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)

        resized = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR → RGB → PIL → ImageTk
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil_img)

        # Center on canvas
        x = (cw - disp_w) // 2
        y = (ch - disp_h) // 2

        if self._canvas_image_id is None:
            self._canvas.delete("placeholder")
            self._canvas_image_id = self._canvas.create_image(x, y, anchor="nw", image=tk_img)
        else:
            self._canvas.coords(self._canvas_image_id, x, y)
            self._canvas.itemconfig(self._canvas_image_id, image=tk_img)
        self._canvas._tk_image = tk_img  # Prevent GC

    def _refresh_stats(self):
        """Update all stats labels."""
        stats = self._last_stats

        # Header badges
        fps = stats.fps
        fps_color = PALETTE["success"] if fps >= 25 else (PALETTE["warning"] if fps >= 15 else PALETTE["danger"])
        self._fps_badge.set_value(f"{fps:.0f}", fps_color)

        pose_color = PALETTE["success"] if stats.pose_detected else PALETTE["danger"]
        self._pose_badge.set_value("✓" if stats.pose_detected else "✗", pose_color)

        dbg_on = any(self.pipeline.get_debug_overlays().values())
        self._model_badge.set_value("Holistic+DBG" if dbg_on else "Holistic", PALETTE["accent"])

        # Performance panel
        def fmt_ms(v): return f"{v:.1f} ms" if v > 0 else "—"
        self._perf_labels["_pose_ms"].configure(text=fmt_ms(stats.pose_ms))
        self._perf_labels["_warp_ms"].configure(text=fmt_ms(stats.warp_ms))
        self._perf_labels["_render_ms"].configure(text=fmt_ms(stats.render_ms))

        total_color = PALETTE["success"] if stats.total_ms < 40 else PALETTE["warning"]
        self._perf_labels["_total_ms"].configure(
            text=fmt_ms(stats.total_ms),
            text_color=total_color,
        )

        # Engine info
        self._engine_label.configure(
            text=f"Engine: {stats.engine_method or '—'}  |  MediaPipe Holistic"
        )

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    def on_close(self):
        """Clean shutdown."""
        self._stop_camera()
        self.pipeline.unload_models()
        self.destroy()


def run_app(pipeline: RenderPipeline):
    """Entry point to launch the GUI application."""
    app = VirtualTryOnApp(pipeline)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
