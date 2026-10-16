"""
Mission Planner GUI  —  UAV B-Spline MILP Navigation
=====================================================
Click-and-drag on the map to place obstacles, configure
the drone, then generate and run the simulation.

Run:  python mission_planner.py
"""

import tkinter as tk
from tkinter import messagebox
import subprocess, sys, os, threading

# ─────────────────────────────────────────────────── #
#  WORLD / CANVAS SETTINGS                             #
# ─────────────────────────────────────────────────── #
WORLD_XLO, WORLD_XHI = -3000, 3000
WORLD_YLO, WORLD_YHI = -3000, 3000
CANVAS_W = CANVAS_H  = 660
Z_CEILING = 25.0

def w2c(wx, wy):
    cx = (wx - WORLD_XLO) / (WORLD_XHI - WORLD_XLO) * CANVAS_W
    cy = CANVAS_H - (wy - WORLD_YLO) / (WORLD_YHI - WORLD_YLO) * CANVAS_H
    return cx, cy

def c2w(cx, cy):
    wx = WORLD_XLO + cx / CANVAS_W * (WORLD_XHI - WORLD_XLO)
    wy = WORLD_YLO + (CANVAS_H - cy) / CANVAS_H * (WORLD_YHI - WORLD_YLO)
    return wx, wy

# ─────────────────────────────────────────────────── #
#  DESIGN TOKENS                                        #
# ─────────────────────────────────────────────────── #
BG       = "#0d1117"
PANEL    = "#161b22"
CARD     = "#1c2128"
BORDER   = "#30363d"
ACCENT_G = "#2ea043"   # green  – run button
ACCENT_B = "#1f6feb"   # blue   – visualiser / apply
ACCENT_R = "#da3633"   # red    – delete / clear
TXT      = "#e6edf3"
TXT2     = "#8b949e"
SEL      = "#ffa657"   # orange – selected obstacle

C_TALL_F = "#0d2a4a";  C_TALL_E = "#388bfd"
C_LOW_F  = "#0d2d1a";  C_LOW_E  = "#3fb950"
C_NFZ_F  = "#2d0d12";  C_NFZ_E  = "#f85149"
C_START  = "#3fb950";  C_GOAL   = "#f0883e"

FONTS = {
    "title":  ("Segoe UI", 12, "bold"),
    "label":  ("Segoe UI",  9),
    "small":  ("Segoe UI",  8),
    "bold":   ("Segoe UI",  9, "bold"),
    "mono":   ("Consolas",  9),
    "btn":    ("Segoe UI", 10, "bold"),
    "btnSm":  ("Segoe UI",  9, "bold"),
}

# ─────────────────────────────────────────────────── #
#  DATA MODEL                                           #
# ─────────────────────────────────────────────────── #
class Obstacle:
    _ctr = 0
    def __init__(self, xl, xu, yl, yu, height=200.0, eta=80.0, otype="tall"):
        Obstacle._ctr += 1
        self.id   = Obstacle._ctr
        self.xl   = min(xl, xu);  self.xu = max(xl, xu)
        self.yl   = min(yl, yu);  self.yu = max(yl, yu)
        self.height = height;     self.eta = eta
        self.otype  = otype        # "tall" | "low" | "nfz"
        self.cid = self.lid = None

    @property
    def is_nfz(self): return self.otype == "nfz"

    def tag(self):
        return {"tall":"TALL","low":"LOW","nfz":"NFZ"}[self.otype] + f"-{self.id}"

    def to_code(self):
        if self.is_nfz:
            return (f"    planner.add_no_fly_zone("
                    f"{self.xl:.0f},{self.xu:.0f},"
                    f"{self.yl:.0f},{self.yu:.0f},gamma={self.eta:.1f})")
        return (f"    planner.add_cuboid_obstacle("
                f"{self.xl:.0f},{self.xu:.0f},"
                f"{self.yl:.0f},{self.yu:.0f},"
                f"0,{self.height:.1f},eta={self.eta:.1f})")


# ─────────────────────────────────────────────────── #
#  HELPER WIDGETS                                       #
# ─────────────────────────────────────────────────── #
def make_sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=6)

def make_label(parent, text, fg=TXT2, font="label", anchor="w", **kw):
    return tk.Label(parent, text=text, bg=PANEL, fg=fg,
                    font=FONTS[font], anchor=anchor, **kw)

def make_entry(parent, var, width=13):
    return tk.Entry(parent, textvariable=var, bg=CARD, fg=TXT,
                    insertbackground=TXT, relief="flat", font=FONTS["mono"],
                    width=width, highlightthickness=1,
                    highlightbackground=BORDER, highlightcolor=ACCENT_B)

def make_btn(parent, text, cmd, bg=ACCENT_G, fg="white", font="btn", pady=7, **kw):
    return tk.Button(parent, text=text, command=cmd,
                     bg=bg, fg=fg, activebackground=bg,
                     activeforeground=fg, relief="flat",
                     font=FONTS[font], pady=pady,
                     cursor="hand2", **kw)

def row_pair(parent, label, var):
    f = tk.Frame(parent, bg=PANEL)
    f.pack(fill="x", padx=12, pady=2)
    make_label(f, label, width=16).pack(side="left")
    make_entry(f, var).pack(side="left", padx=(4,0))
    return f

def section_hdr(parent, text):
    f = tk.Frame(parent, bg=PANEL)
    f.pack(fill="x", padx=12, pady=(10,0))
    tk.Label(f, text=text, bg=PANEL, fg=ACCENT_B,
             font=FONTS["bold"]).pack(side="left")
    tk.Frame(f, bg=BORDER, height=1).place(relx=0, rely=0.9, relwidth=1)
    make_sep(parent)


# ─────────────────────────────────────────────────── #
#  MAIN APP                                             #
# ─────────────────────────────────────────────────── #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UAV Mission Planner")
        self.configure(bg=BG)
        self.minsize(1100, 720)

        # state
        self.obstacles  = []
        self.selected   = None
        self.draw_mode  = tk.StringVar(value="tall")
        self._drag0     = None
        self._drag_rect = None

        # sidebar vars
        self.sv_start  = tk.StringVar(value="2500, -1400, 12")
        self.sv_goal   = tk.StringVar(value="-2200, 2000, 12")
        self.sv_vmax   = tk.DoubleVar(value=60.0)
        self.sv_wind   = tk.BooleanVar(value=True)
        self.sv_dyn    = tk.BooleanVar(value=False)
        self.sv_xl = tk.StringVar(); self.sv_xu = tk.StringVar()
        self.sv_yl = tk.StringVar(); self.sv_yu = tk.StringVar()
        self.sv_h  = tk.StringVar(); self.sv_e  = tk.StringVar()
        self.sv_ot = tk.StringVar(value="tall")

        self._build()
        self._grid()
        self._sg()

    # ── Layout ────────────────────────────────────── #
    def _build(self):
        # root: two columns
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=340)
        self.grid_rowconfigure(0, weight=1)

        # ── LEFT: map ─────────────────────────────── #
        lf = tk.Frame(self, bg=BG)
        lf.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        lf.grid_rowconfigure(1, weight=1)

        # title bar
        hdr = tk.Frame(lf, bg=BG)
        hdr.grid(row=0, column=0, sticky="ew", pady=(0,6))
        tk.Label(hdr, text="Mission Map", bg=BG, fg=TXT,
                 font=FONTS["title"]).pack(side="left")
        tk.Label(hdr, text="  Left-drag to draw  |  Right-click to select",
                 bg=BG, fg=TXT2, font=FONTS["small"]).pack(side="left", padx=8)

        # canvas
        cf = tk.Frame(lf, bg=BORDER, bd=1)
        cf.grid(row=1, column=0, sticky="nsew")
        self.cv = tk.Canvas(cf, width=CANVAS_W, height=CANVAS_H,
                            bg="#0a0f1a", highlightthickness=0, cursor="crosshair")
        self.cv.pack(expand=True, fill="both")
        self.cv.bind("<ButtonPress-1>",   self._md)
        self.cv.bind("<B1-Motion>",       self._mm)
        self.cv.bind("<ButtonRelease-1>", self._mu)
        self.cv.bind("<Button-3>",        self._rc)

        # toolbar
        tb = tk.Frame(lf, bg=BG)
        tb.grid(row=2, column=0, sticky="ew", pady=(8,0))
        tk.Label(tb, text="Draw:", bg=BG, fg=TXT2,
                 font=FONTS["bold"]).pack(side="left", padx=(0,6))

        mode_defs = [
            ("Tall Building (200m)",  "tall", C_TALL_E),
            ("Low Hurdle (15m)",      "low",  C_LOW_E),
            ("No-Fly Zone",           "nfz",  C_NFZ_E),
        ]
        for label, val, col in mode_defs:
            rb = tk.Radiobutton(tb, text=label, variable=self.draw_mode, value=val,
                                bg=CARD, fg=col, selectcolor=BG,
                                activebackground=CARD, activeforeground=col,
                                font=FONTS["bold"], relief="flat",
                                padx=10, pady=5, indicatoron=False, bd=1,
                                cursor="hand2")
            rb.pack(side="left", padx=3)

        # legend
        for col, txt in [(C_TALL_E,"=Tall"), (C_LOW_E,"=Low"), (C_NFZ_E,"=NFZ"),
                          (C_START,"=Start"), (C_GOAL,"=Goal")]:
            tk.Label(tb, text=f"  ●", bg=BG, fg=col,
                     font=FONTS["bold"]).pack(side="right")
            tk.Label(tb, text=txt, bg=BG, fg=TXT2,
                     font=FONTS["small"]).pack(side="right")

        # ── RIGHT: sidebar (3-row grid) ────────────── #
        # row0 = scrollable config area  (expands)
        # row1 = obstacle list           (fixed ~160px)
        # row2 = action buttons          (fixed, always visible)
        sf = tk.Frame(self, bg=PANEL, width=340)
        sf.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=10)
        sf.grid_propagate(False)
        sf.grid_rowconfigure(0, weight=3)   # config area
        sf.grid_rowconfigure(1, weight=2)   # list
        sf.grid_rowconfigure(2, weight=0)   # buttons ALWAYS VISIBLE
        sf.grid_columnconfigure(0, weight=1)

        # ── Config area (scrollable) ────────────────── #
        cfg_outer = tk.Frame(sf, bg=PANEL)
        cfg_outer.grid(row=0, column=0, sticky="nsew")
        cfg_outer.grid_rowconfigure(0, weight=1)
        cfg_outer.grid_columnconfigure(0, weight=1)

        cfg_cv = tk.Canvas(cfg_outer, bg=PANEL, highlightthickness=0)
        cfg_cv.grid(row=0, column=0, sticky="nsew")
        cfg_sb = tk.Scrollbar(cfg_outer, orient="vertical",
                              command=cfg_cv.yview)
        cfg_sb.grid(row=0, column=1, sticky="ns")
        cfg_cv.configure(yscrollcommand=cfg_sb.set)

        self.cfg = tk.Frame(cfg_cv, bg=PANEL)
        cfg_cv.create_window((0,0), window=self.cfg, anchor="nw", width=336)
        self.cfg.bind("<Configure>",
                      lambda e: cfg_cv.configure(
                          scrollregion=cfg_cv.bbox("all")))

        self._build_config(self.cfg)

        # ── Obstacle List ──────────────────────────── #
        lf2 = tk.Frame(sf, bg=PANEL)
        lf2.grid(row=1, column=0, sticky="nsew", padx=2)
        lf2.grid_rowconfigure(1, weight=1)
        lf2.grid_columnconfigure(0, weight=1)

        hdr2 = tk.Frame(lf2, bg=PANEL)
        hdr2.grid(row=0, column=0, sticky="ew", padx=10, pady=(8,4))
        tk.Label(hdr2, text="Obstacle List", bg=PANEL, fg=ACCENT_B,
                 font=FONTS["bold"]).pack(side="left")
        make_btn(hdr2, "Clear All", self._clear_all,
                 bg=ACCENT_R, font="btnSm", pady=3).pack(side="right")
        make_btn(hdr2, "Load Default", self._load_default,
                 bg=CARD, fg=TXT, font="btnSm", pady=3).pack(side="right", padx=(0,6))

        lbox_f = tk.Frame(lf2, bg=CARD)
        lbox_f.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,8))
        lbox_f.grid_rowconfigure(0, weight=1)
        lbox_f.grid_columnconfigure(0, weight=1)
        lsb = tk.Scrollbar(lbox_f, orient="vertical", bg=BG)
        self.lb = tk.Listbox(lbox_f, bg=CARD, fg=TXT,
                             selectbackground=ACCENT_B, selectforeground="white",
                             font=FONTS["mono"], relief="flat", bd=0,
                             activestyle="none", yscrollcommand=lsb.set)
        lsb.config(command=self.lb.yview)
        self.lb.grid(row=0, column=0, sticky="nsew")
        lsb.grid(row=0, column=1, sticky="ns")
        self.lb.bind("<<ListboxSelect>>", self._lb_sel)

        # ── Action Buttons (ALWAYS VISIBLE at bottom) ─ #
        btn_f = tk.Frame(sf, bg=BG)
        btn_f.grid(row=2, column=0, sticky="ew")
        btn_f.grid_columnconfigure(0, weight=1)

        sep = tk.Frame(btn_f, bg=BORDER, height=2)
        sep.grid(row=0, column=0, sticky="ew")

        inner = tk.Frame(btn_f, bg=BG)
        inner.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        inner.grid_columnconfigure(0, weight=1)

        make_btn(inner, "  Generate & Run Simulation",
                 self._run_sim, bg=ACCENT_G,
                 font="btn", pady=10).grid(row=0, column=0, sticky="ew", pady=(0,6))

        make_btn(inner, "  Open Live Visualizer",
                 self._run_vis, bg=ACCENT_B,
                 font="btnSm", pady=7).grid(row=1, column=0, sticky="ew")

        self.status = tk.Label(inner,
                               text="Ready. Draw obstacles then run.",
                               bg=BG, fg=TXT2, font=FONTS["small"],
                               anchor="w", wraplength=310)
        self.status.grid(row=2, column=0, sticky="ew", pady=(8,0))

    def _build_config(self, p):
        """Scrollable config content."""
        tk.Label(p, text="Mission Parameters",
                 bg=PANEL, fg=TXT, font=FONTS["title"]).pack(
                 anchor="w", padx=12, pady=(12,6))

        # Flight params
        section_hdr(p, "Drone & Flight")
        row_pair(p, "Start  x,y,z (m)", self.sv_start)
        row_pair(p, "Goal   x,y,z (m)", self.sv_goal)

        # Speed slider
        sf = tk.Frame(p, bg=PANEL)
        sf.pack(fill="x", padx=12, pady=2)
        make_label(sf, "Max Speed (m/s)", width=16).pack(side="left")
        self._slbl = tk.Label(sf, text="60", bg=PANEL, fg=SEL,
                              font=FONTS["bold"], width=4)
        self._slbl.pack(side="right")
        tk.Scale(sf, variable=self.sv_vmax, from_=10, to=120,
                 orient="horizontal", bg=PANEL, fg=TXT2,
                 troughcolor=CARD, activebackground=SEL,
                 highlightthickness=0, showvalue=False, length=130,
                 command=lambda v: self._slbl.config(text=f"{float(v):.0f}")
                 ).pack(side="left", padx=4)

        # Toggles
        for text, var in [("Enable Dryden Wind Turbulence", self.sv_wind),
                           ("Enable Intruder UAV",          self.sv_dyn)]:
            cf = tk.Frame(p, bg=PANEL)
            cf.pack(fill="x", padx=12, pady=1)
            tk.Checkbutton(cf, text=text, variable=var, bg=PANEL, fg=TXT,
                           selectcolor=CARD, activebackground=PANEL,
                           activeforeground=TXT, font=FONTS["label"],
                           cursor="hand2").pack(side="left")

        make_sep(p)

        # Selected obstacle editor
        section_hdr(p, "Edit Selected Obstacle")

        self._sel_lbl = make_label(p, "(no obstacle selected)",
                                   fg=TXT2, font="small")
        self._sel_lbl.pack(anchor="w", padx=12)

        # Type radio buttons
        tr = tk.Frame(p, bg=PANEL)
        tr.pack(fill="x", padx=12, pady=4)
        make_label(tr, "Type", width=10).pack(side="left")
        for val, col, txt in [("tall", C_TALL_E, "Tall"),
                               ("low",  C_LOW_E,  "Low"),
                               ("nfz",  C_NFZ_E,  "NFZ")]:
            tk.Radiobutton(tr, text=txt, variable=self.sv_ot, value=val,
                           bg=CARD, fg=col, selectcolor=BG,
                           activebackground=CARD, font=FONTS["bold"],
                           relief="flat", padx=8, pady=3,
                           indicatoron=False, cursor="hand2",
                           command=self._ot_changed).pack(side="left", padx=2)

        for lbl, sv in [("X Low  (m)", self.sv_xl), ("X High (m)", self.sv_xu),
                         ("Y Low  (m)", self.sv_yl), ("Y High (m)", self.sv_yu),
                         ("Height (m)", self.sv_h),  ("Safety η (m)", self.sv_e)]:
            row_pair(p, lbl, sv)

        # Apply / Delete
        ab = tk.Frame(p, bg=PANEL)
        ab.pack(fill="x", padx=12, pady=(6,4))
        make_btn(ab, "Apply", self._apply_edit, bg=ACCENT_B,
                 font="btnSm", pady=5).pack(side="left", padx=(0,6))
        make_btn(ab, "Delete", self._del_sel, bg=ACCENT_R,
                 font="btnSm", pady=5).pack(side="left")

        make_sep(p)

    # ── Canvas ─────────────────────────────────────── #
    def _grid(self):
        self.cv.delete("grid")
        for i in range(13):
            x = i * CANVAS_W / 12
            self.cv.create_line(x, 0, x, CANVAS_H, fill="#162030", tags="grid")
        for j in range(13):
            y = j * CANVAS_H / 12
            self.cv.create_line(0, y, CANVAS_W, y, fill="#162030", tags="grid")
        cx, cy = w2c(0, 0)
        self.cv.create_line(cx, 0, cx, CANVAS_H, fill="#243040", width=2, tags="grid")
        self.cv.create_line(0, cy, CANVAS_W, cy, fill="#243040", width=2, tags="grid")
        for v in [-2000,-1000,0,1000,2000]:
            px, _ = w2c(v, 0);  _,py = w2c(0, v)
            self.cv.create_text(px, cy+12, text=str(v), fill=TXT2,
                                font=FONTS["small"], tags="grid")
            self.cv.create_text(cx+24, py, text=str(v), fill=TXT2,
                                font=FONTS["small"], tags="grid")

    def _sg(self):
        self.cv.delete("sg")
        try:
            sx,sy,sz = [float(v.strip()) for v in self.sv_start.get().split(",")]
            gx,gy,gz = [float(v.strip()) for v in self.sv_goal.get().split(",")]
        except Exception: return
        scx,scy = w2c(sx,sy);  gcx,gcy = w2c(gx,gy)
        self.cv.create_line(scx,scy,gcx,gcy,fill="#25303a",width=1,dash=(5,5),tags="sg")
        r=10
        self.cv.create_oval(scx-r,scy-r,scx+r,scy+r,
                            fill=C_START,outline="white",width=2,tags="sg")
        self.cv.create_text(scx,scy,text="S",fill="white",
                            font=FONTS["bold"],tags="sg")
        self.cv.create_text(scx+r+22,scy,text="START",fill=C_START,
                            font=FONTS["small"],tags="sg")
        self.cv.create_oval(gcx-r,gcy-r,gcx+r,gcy+r,
                            fill=C_GOAL,outline="white",width=2,tags="sg")
        self.cv.create_text(gcx,gcy,text="G",fill="white",
                            font=FONTS["bold"],tags="sg")
        self.cv.create_text(gcx+r+20,gcy,text="GOAL",fill=C_GOAL,
                            font=FONTS["small"],tags="sg")

    def _redraw(self, obs):
        for iid in (obs.cid, obs.lid):
            if iid: self.cv.delete(iid)
        x1,y1 = w2c(obs.xl,obs.yl);  x2,y2 = w2c(obs.xu,obs.yu)
        fills  = {"tall":(C_TALL_F,C_TALL_E),"low":(C_LOW_F,C_LOW_E),
                  "nfz":(C_NFZ_F,C_NFZ_E)}
        fill,out = fills[obs.otype]
        ol = SEL if (self.selected is obs) else out
        ow = 3  if (self.selected is obs) else 1.5
        obs.cid = self.cv.create_rectangle(x1,y2,x2,y1,
                                            fill=fill,outline=ol,width=ow)
        obs.lid = self.cv.create_text((x1+x2)/2,(y1+y2)/2,
                                       text=obs.tag(),fill="white",
                                       font=FONTS["small"])
        self.cv.tag_raise("sg")

    # ── Mouse ──────────────────────────────────────── #
    def _md(self, e): self._drag0=(e.x,e.y); self._drag_rect=None
    def _mm(self, e):
        if not self._drag0: return
        if self._drag_rect: self.cv.delete(self._drag_rect)
        c={"tall":C_TALL_E,"low":C_LOW_E,"nfz":C_NFZ_E}[self.draw_mode.get()]
        x0,y0=self._drag0
        self._drag_rect=self.cv.create_rectangle(x0,y0,e.x,e.y,
                                                  outline=c,width=2,dash=(4,4))
    def _mu(self, e):
        if not self._drag0: return
        x0,y0=self._drag0; x1,y1=e.x,e.y
        self._drag0=None
        if self._drag_rect: self.cv.delete(self._drag_rect); self._drag_rect=None
        if abs(x1-x0)<8 or abs(y1-y0)<8: return
        wx0,wy0=c2w(x0,y0);  wx1,wy1=c2w(x1,y1)
        m=self.draw_mode.get()
        defs={"tall":(200.0,80.0),"low":(15.0,5.0),"nfz":(0.0,30.0)}
        h,eta=defs[m]
        obs=Obstacle(wx0,wx1,wy0,wy1,h,eta,m)
        self.obstacles.append(obs)
        self._redraw(obs); self._upd_lb(); self._sel(obs)

    def _rc(self, e):
        items=self.cv.find_overlapping(e.x-5,e.y-5,e.x+5,e.y+5)
        for item in reversed(items):
            for obs in self.obstacles:
                if obs.cid==item: self._sel(obs); return
        self._desel()

    # ── Selection ──────────────────────────────────── #
    def _sel(self, obs):
        prev=self.selected; self.selected=obs
        if prev and prev is not obs: self._redraw(prev)
        self._redraw(obs); self._pop_sidebar(obs)
        for i,o in enumerate(self.obstacles):
            if o is obs:
                self.lb.selection_clear(0,tk.END)
                self.lb.selection_set(i); self.lb.see(i); break

    def _desel(self):
        prev=self.selected; self.selected=None
        if prev: self._redraw(prev)
        self._sel_lbl.config(text="(no obstacle selected)")

    def _pop_sidebar(self, obs):
        self._sel_lbl.config(text=f"Editing:  {obs.tag()}", fg=TXT)
        self.sv_ot.set(obs.otype)
        self.sv_xl.set(f"{obs.xl:.0f}"); self.sv_xu.set(f"{obs.xu:.0f}")
        self.sv_yl.set(f"{obs.yl:.0f}"); self.sv_yu.set(f"{obs.yu:.0f}")
        self.sv_h.set(f"{obs.height:.1f}"); self.sv_e.set(f"{obs.eta:.1f}")

    def _ot_changed(self):
        if not self.selected: return
        defs={"tall":(200.0,80.0),"low":(15.0,5.0),"nfz":(0.0,30.0)}
        h,e=defs[self.sv_ot.get()]
        self.sv_h.set(f"{h:.1f}"); self.sv_e.set(f"{e:.1f}")

    def _apply_edit(self):
        if not self.selected: return
        obs=self.selected
        try:
            obs.otype=self.sv_ot.get()
            obs.xl=float(self.sv_xl.get()); obs.xu=float(self.sv_xu.get())
            obs.yl=float(self.sv_yl.get()); obs.yu=float(self.sv_yu.get())
            obs.height=float(self.sv_h.get()); obs.eta=float(self.sv_e.get())
        except ValueError:
            messagebox.showerror("Error","All fields must be numbers."); return
        self._redraw(obs); self._upd_lb()
        self.status.config(text=f"Applied: {obs.tag()}")

    def _del_sel(self):
        if not self.selected: return
        obs=self.selected
        self.cv.delete(obs.cid); self.cv.delete(obs.lid)
        self.obstacles.remove(obs); self.selected=None
        self._upd_lb(); self._sel_lbl.config(text="(no obstacle selected)")
        self.status.config(text="Obstacle deleted.")

    def _clear_all(self):
        if not messagebox.askyesno("Clear All","Delete all obstacles?"): return
        for obs in self.obstacles:
            self.cv.delete(obs.cid); self.cv.delete(obs.lid)
        self.obstacles.clear(); self.selected=None; Obstacle._ctr=0
        self._upd_lb(); self.status.config(text="All obstacles cleared.")

    def _lb_sel(self, e):
        sel=self.lb.curselection()
        if sel and sel[0]<len(self.obstacles):
            self._sel(self.obstacles[sel[0]])

    def _upd_lb(self):
        self.lb.delete(0,tk.END)
        icons={"tall":"[T]","low":"[L]","nfz":"[N]"}
        for obs in self.obstacles:
            dim=f" x[{obs.xl:.0f}..{obs.xu:.0f}] y[{obs.yl:.0f}..{obs.yu:.0f}]"
            self.lb.insert(tk.END, f"{icons[obs.otype]} {obs.tag()}{dim}")

    # ── Load Default ───────────────────────────────── #
    def _load_default(self):
        self._clear_all()
        self.sv_start.set("2500, -1400, 12"); self.sv_goal.set("-2200, 2000, 12")
        self.sv_vmax.set(60.0); self._slbl.config(text="60")
        raw=[
            (1800,2200,-2100,-1700,200.0,80.0,"tall"),
            (1800,2200, -800, -400,200.0,80.0,"tall"),
            (1350,1650,-2000,  200, 15.0, 5.0,"low"),
            ( 800,1200,-1200, -800,200.0,80.0,"tall"),
            ( 800,1200, -250,  150,200.0,80.0,"tall"),
            ( 250, 550,-1200, 1000, 15.0, 5.0,"low"),
            (-400,   0,  -50,  200,200.0,80.0,"tall"),
            (-400,   0,  520,  820,200.0,80.0,"tall"),
            (-950,-650, -300, 1700, 15.0, 5.0,"low"),
            (-1800,-1400,1000,1300,200.0,80.0,"tall"),
            (-1800,-1400,1700,2000,200.0,80.0,"tall"),
        ]
        nfzs=[( 500,1300,-2300,-1600,30.0),
              (-500, 300,  700, 1200,30.0),
              (-2100,-1200,1600,2100,30.0)]
        for xl,xu,yl,yu,h,e,t in raw:
            obs=Obstacle(xl,xu,yl,yu,h,e,t); self.obstacles.append(obs); self._redraw(obs)
        for xl,xu,yl,yu,g in nfzs:
            obs=Obstacle(xl,xu,yl,yu,0.0,g,"nfz"); self.obstacles.append(obs); self._redraw(obs)
        self._upd_lb(); self._sg()
        self.status.config(text="Default mixed-height canyon loaded.")

    # ── Scenario Generator ─────────────────────────── #
    def _gen_scenario(self):
        try:
            sx,sy,sz=[float(v.strip()) for v in self.sv_start.get().split(",")]
            gx,gy,gz=[float(v.strip()) for v in self.sv_goal.get().split(",")]
        except Exception:
            messagebox.showerror("Error","Start/Goal must be 3 comma-separated numbers.")
            return False
        vmax=self.sv_vmax.get()
        blds=[o for o in self.obstacles if not o.is_nfz]
        nfzs=[o for o in self.obstacles if o.is_nfz]
        lines=[
            '"""Custom Scenario -- generated by Mission Planner GUI"""',
            'import numpy as np','from dynamic_obstacle import DynamicObstacle',
            '','Z_CEILING = 25.0','',
            'def build_scenario(planner):',
            f'    x_s=np.array([{sx:.1f},{sy:.1f},{sz:.1f}])',
            f'    v_s=np.array([0.0,0.0,0.0])',
            f'    a_s=np.array([0.0,0.0,0.0])',
            f'    x_f=np.array([{gx:.1f},{gy:.1f},{gz:.1f}])',
            f'    planner.v_max={vmax:.1f}',''
        ]
        if blds:
            lines.append('    # Buildings')
            for o in blds: lines.append(o.to_code())
            lines.append('')
        if nfzs:
            lines.append('    # No-Fly Zones')
            for o in nfzs: lines.append(o.to_code())
            lines.append('')
        lines+=['    return x_s,v_s,a_s,x_f','','',
                'def create_dynamic_obstacle():']
        if self.sv_dyn.get():
            lines+=['    return DynamicObstacle(',
                    '        pos0=[2200.0,-2000.0,12.0],',
                    '        velocity=[-50.0,90.0,0.0],',
                    '        radius=200.0,safety_margin=50.0)']
        else: lines+=['    return None']
        path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"scenario.py")
        with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines)+"\n")
        return True

    # ── Spinner animation while simulation runs ──────── #
    def _start_spinner(self):
        self._spinner_running = True
        self._spinner_dots = 0
        self._spin_step()

    def _spin_step(self):
        if not self._spinner_running:
            return
        dots = '.' * (self._spinner_dots % 4)
        elapsed = getattr(self, '_sim_elapsed', 0)
        self._sim_elapsed = elapsed + 1
        self.status.config(
            text=f"Running simulation{dots}  ({self._sim_elapsed}s elapsed)")
        self._spinner_dots += 1
        self._spinner_id = self.after(1000, self._spin_step)

    def _stop_spinner(self):
        self._spinner_running = False
        if hasattr(self, '_spinner_id'):
            self.after_cancel(self._spinner_id)

    # ── Threaded simulation runner ───────────────────── #
    def _run_sim(self):
        self._sg()
        self.status.config(text="Writing scenario.py ...")
        self.update()
        if not self._gen_scenario():
            return

        # Disable the run button so it can't be clicked twice
        self._run_btn.config(state="disabled", text="Running...")
        self._sim_elapsed = 0
        self._start_spinner()

        d = os.path.dirname(os.path.abspath(__file__))

        def worker():
            try:
                r = subprocess.run(
                    [sys.executable, "simulation.py"],
                    cwd=d, timeout=600)
                msg = ("Done! Graphs saved. Click 'Open Live Visualizer'."
                       if r.returncode == 0
                       else f"Simulation failed (exit {r.returncode}). Check terminal.")
            except subprocess.TimeoutExpired:
                msg = "Timed out (>10 min)."
            except Exception as e:
                msg = f"Error: {e}"
            # schedule UI update back on the main thread
            self.after(0, lambda: self._sim_done(msg))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _sim_done(self, msg):
        self._stop_spinner()
        self._run_btn.config(state="normal", text="  Generate & Run Simulation")
        self.status.config(text=msg)

    def _run_vis(self):
        d = os.path.dirname(os.path.abspath(__file__))
        self.status.config(text="Launching visualizer ...")
        self.update()
        subprocess.Popen([sys.executable, "visualizer.py"], cwd=d)


if __name__=="__main__":
    app=App()
    app.after(150, app._load_default)
    app.mainloop()
