import argparse
import math
import tkinter as tk
from tkinter import ttk

from runtime import DemoRuntime, run_headless_smoke


CANVAS_W = 720
CANVAS_H = 360
TRACK_Y = 250
TRACK_HALF_WIDTH = 240
CART_W = 90
CART_H = 34
POLE_PX = 130
HISTORY_H = 120
STATE_KEYS = ('x', 'theta', 'x_dot', 'theta_dot')


class CartpoleDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Cartpole SD-MCTS Demo')
        self.runtime = DemoRuntime()
        self.running = False
        self.tick_ms = 80
        self.snapshot = self.runtime.get_snapshot()

        self.seed_var = tk.IntVar(value=self.snapshot['seed'])
        self.mode_var = tk.StringVar(value=self.snapshot['mode'])
        self.sim_var = tk.IntVar(value=self.snapshot['num_simulations'])
        self.auto_var = tk.BooleanVar(value=False)

        self.param_vars = {
            'gravity': tk.DoubleVar(value=self.snapshot['params']['gravity']),
            'masscart': tk.DoubleVar(value=self.snapshot['params']['masscart']),
            'masspole': tk.DoubleVar(value=self.snapshot['params']['masspole']),
            'length': tk.DoubleVar(value=self.snapshot['params']['length']),
        }

        self.status_var = tk.StringVar(value='Ready')
        self.header_var = tk.StringVar()
        self.state_text = tk.StringVar()
        self.params_text = tk.StringVar()
        self.summary_text = tk.StringVar()
        self.trace_text = tk.StringVar()
        self.selected_cfg_text = tk.StringVar()

        self._build_ui()
        self._refresh_all()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky='nsew')
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=3)
        left.rowconfigure(1, weight=1)

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg='white', highlightthickness=1, highlightbackground='#bbbbbb')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        self.history_canvas = tk.Canvas(left, width=CANVAS_W, height=HISTORY_H, bg='white', highlightthickness=1, highlightbackground='#bbbbbb')
        self.history_canvas.grid(row=1, column=0, sticky='ew', pady=(8, 0))

        header = ttk.Label(left, textvariable=self.header_var, font=('TkDefaultFont', 11, 'bold'))
        header.grid(row=2, column=0, sticky='ew', pady=(8, 4))

        metrics = ttk.Frame(left)
        metrics.grid(row=3, column=0, sticky='ew')
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        metrics.columnconfigure(2, weight=1)
        ttk.Label(metrics, textvariable=self.state_text, justify='left').grid(row=0, column=0, sticky='nw')
        ttk.Label(metrics, textvariable=self.params_text, justify='left').grid(row=0, column=1, sticky='nw')
        ttk.Label(metrics, textvariable=self.summary_text, justify='left').grid(row=0, column=2, sticky='nw')

        control = ttk.LabelFrame(right, text='Controls', padding=8)
        control.grid(row=0, column=0, sticky='ew')
        control.columnconfigure(1, weight=1)

        seed_values = sorted(self.runtime.checkpoints)
        ttk.Label(control, text='Checkpoint seed').grid(row=0, column=0, sticky='w')
        seed_combo = ttk.Combobox(control, textvariable=self.seed_var, values=seed_values, state='readonly', width=10)
        seed_combo.grid(row=0, column=1, sticky='ew', padx=(8, 0))
        seed_combo.bind('<<ComboboxSelected>>', lambda _e: self._load_seed())

        quick_seed_row = ttk.Frame(control)
        quick_seed_row.grid(row=0, column=2, sticky='e', padx=(8, 0))
        for idx, seed in enumerate(seed_values):
            ttk.Button(quick_seed_row, text=f'S{seed}', width=5, command=lambda s=seed: self._quick_seed(s)).grid(row=0, column=idx, padx=(0 if idx == 0 else 4, 0))

        ttk.Label(control, text='Mode').grid(row=1, column=0, sticky='w', pady=(6, 0))
        mode_combo = ttk.Combobox(control, textvariable=self.mode_var, values=['sd_mcts', 'fast', 'heuristic'], state='readonly', width=12)
        mode_combo.grid(row=1, column=1, sticky='ew', padx=(8, 0), pady=(6, 0))
        mode_combo.bind('<<ComboboxSelected>>', lambda _e: self._set_mode())

        ttk.Label(control, text='SD-MCTS sims').grid(row=2, column=0, sticky='w', pady=(6, 0))
        sim_scale = ttk.Scale(control, from_=5, to=200, variable=self.sim_var, command=lambda _v: self._set_sims())
        sim_scale.grid(row=2, column=1, sticky='ew', padx=(8, 0), pady=(6, 0))

        button_row = ttk.Frame(control)
        button_row.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        for idx in range(4):
            button_row.columnconfigure(idx, weight=1)
        ttk.Button(button_row, text='Step', command=self._step_once).grid(row=0, column=0, sticky='ew')
        ttk.Button(button_row, text='Reset', command=self._reset_episode).grid(row=0, column=1, sticky='ew', padx=4)
        ttk.Button(button_row, text='Restore params', command=self._restore_params).grid(row=0, column=2, sticky='ew')
        ttk.Checkbutton(button_row, text='Auto-run', variable=self.auto_var, command=self._toggle_run).grid(row=0, column=3, sticky='ew', padx=(4, 0))

        param_box = ttk.LabelFrame(right, text='Counterfactual parameters', padding=8)
        param_box.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        param_specs = {
            'gravity': (5.0, 15.0),
            'masscart': (0.5, 2.0),
            'masspole': (0.03, 0.3),
            'length': (0.2, 1.0),
        }
        for idx, (name, (lo, hi)) in enumerate(param_specs.items()):
            ttk.Label(param_box, text=name).grid(row=idx, column=0, sticky='w')
            scale = ttk.Scale(param_box, from_=lo, to=hi, variable=self.param_vars[name])
            scale.grid(row=idx, column=1, sticky='ew', padx=(8, 8))
            ttk.Label(param_box, textvariable=self.param_vars[name], width=7).grid(row=idx, column=2, sticky='e')
        param_box.columnconfigure(1, weight=1)
        ttk.Button(param_box, text='Apply to world model', command=self._apply_params).grid(row=len(param_specs), column=0, columnspan=3, sticky='ew', pady=(8, 0))

        cfg_box = ttk.LabelFrame(right, text='Checkpoint / training metadata', padding=8)
        cfg_box.grid(row=2, column=0, sticky='ew', pady=(10, 0))
        ttk.Label(cfg_box, textvariable=self.selected_cfg_text, justify='left').grid(row=0, column=0, sticky='w')

        trace_box = ttk.LabelFrame(right, text='Search / action trace', padding=8)
        trace_box.grid(row=3, column=0, sticky='nsew', pady=(10, 0))
        trace_box.columnconfigure(0, weight=1)
        trace_box.rowconfigure(1, weight=1)
        self.bar_canvas = tk.Canvas(trace_box, width=420, height=150, bg='white', highlightthickness=1, highlightbackground='#cccccc')
        self.bar_canvas.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        trace_label = ttk.Label(trace_box, textvariable=self.trace_text, justify='left')
        trace_label.grid(row=1, column=0, sticky='nw')

        status = ttk.Label(right, textvariable=self.status_var, relief='sunken', anchor='w')
        status.grid(row=4, column=0, sticky='ew', pady=(10, 0))

    def _load_seed(self):
        self.runtime.load_checkpoint(self.seed_var.get())
        self.snapshot = self.runtime.get_snapshot()
        self._sync_controls_from_snapshot()
        self._refresh_all('Loaded checkpoint')

    def _quick_seed(self, seed):
        self.seed_var.set(seed)
        self._load_seed()

    def _set_mode(self):
        self.runtime.set_mode(self.mode_var.get())
        self.snapshot = self.runtime.get_snapshot()
        self._refresh_all('Switched mode')

    def _set_sims(self):
        self.runtime.set_num_simulations(self.sim_var.get())
        self.snapshot = self.runtime.get_snapshot()
        self._refresh_all('Updated simulations')

    def _step_once(self):
        self.snapshot = self.runtime.step_decision()
        self._refresh_all('Stepped controller')

    def _reset_episode(self):
        self.snapshot = self.runtime.reset_episode()
        self._refresh_all('Reset episode')

    def _restore_params(self):
        self.snapshot = self.runtime.restore_checkpoint_params()
        self._sync_param_vars()
        self._refresh_all('Restored checkpoint parameters')

    def _apply_params(self):
        self.snapshot = self.runtime.apply_counterfactual(
            self.param_vars['gravity'].get(),
            self.param_vars['masscart'].get(),
            self.param_vars['masspole'].get(),
            self.param_vars['length'].get(),
        )
        self._refresh_all('Applied counterfactual parameters')

    def _toggle_run(self):
        self.running = self.auto_var.get()
        if self.running:
            self._auto_step()
        else:
            self.status_var.set('Paused')

    def _auto_step(self):
        if not self.running:
            return
        self.snapshot = self.runtime.step_decision()
        if self.snapshot['done']:
            self.running = False
            self.auto_var.set(False)
            self._refresh_all('Episode finished')
            return
        self._refresh_all('Auto-running')
        self.root.after(self.tick_ms, self._auto_step)

    def _sync_controls_from_snapshot(self):
        self.seed_var.set(self.snapshot['seed'])
        self.mode_var.set(self.snapshot['mode'])
        self.sim_var.set(self.snapshot['num_simulations'])
        self._sync_param_vars()

    def _sync_param_vars(self):
        params = self.snapshot['params']
        for key, var in self.param_vars.items():
            var.set(params[key])

    def _refresh_all(self, status=None):
        self._draw_cartpole()
        self._draw_history()
        self._draw_action_bars()
        self._update_texts()
        if status:
            self.status_var.set(status)

    def _draw_cartpole(self):
        self.canvas.delete('all')
        self.canvas.create_line(40, TRACK_Y + CART_H / 2, CANVAS_W - 40, TRACK_Y + CART_H / 2, fill='#666666', width=3)

        state = self.snapshot['state']
        x, theta = float(state[0]), float(state[1])
        cart_x = CANVAS_W / 2 + max(-2.4, min(2.4, x)) / 2.4 * TRACK_HALF_WIDTH
        cart_y = TRACK_Y
        left = cart_x - CART_W / 2
        top = cart_y - CART_H / 2
        right = cart_x + CART_W / 2
        bottom = cart_y + CART_H / 2
        self.canvas.create_rectangle(left, top, right, bottom, fill='#4c78a8', outline='')
        self.canvas.create_oval(left + 10, bottom - 4, left + 26, bottom + 12, fill='#333333', outline='')
        self.canvas.create_oval(right - 26, bottom - 4, right - 10, bottom + 12, fill='#333333', outline='')

        pivot_x = cart_x
        pivot_y = top + 4
        pole_tip_x = pivot_x + POLE_PX * math.sin(theta)
        pole_tip_y = pivot_y - POLE_PX * math.cos(theta)
        self.canvas.create_line(pivot_x, pivot_y, pole_tip_x, pole_tip_y, fill='#e45756', width=8)
        self.canvas.create_oval(pivot_x - 6, pivot_y - 6, pivot_x + 6, pivot_y + 6, fill='#222222', outline='')
        self.canvas.create_oval(pole_tip_x - 7, pole_tip_y - 7, pole_tip_x + 7, pole_tip_y + 7, fill='#72b7b2', outline='')

        self.canvas.create_text(12, 12, anchor='nw', text=f"mode={self.snapshot['mode']}  seed={self.snapshot['seed']}  done={self.snapshot['done']}", fill='#222222')
        self.canvas.create_text(12, 32, anchor='nw', text=f"last_action={self.snapshot['last_action']:+.2f}  return={self.snapshot['episode_return']:.1f}", fill='#222222')

    def _draw_history(self):
        self.history_canvas.delete('all')
        history = self.snapshot.get('history', {})
        xs = history.get('x', [])
        coss = history.get('cos_theta', [])
        rets = history.get('return', [])
        width = int(self.history_canvas.winfo_width() or CANVAS_W)
        height = int(self.history_canvas.winfo_height() or HISTORY_H)
        self.history_canvas.create_text(10, 8, anchor='nw', text='History: x (blue), cos(theta) (green), return (orange)', fill='#222222')
        self.history_canvas.create_rectangle(40, 24, width - 10, height - 12, outline='#cccccc')
        if len(xs) < 2:
            return

        def make_points(values, vmin, vmax, top, bottom):
            pts = []
            n = max(1, len(values) - 1)
            span = max(vmax - vmin, 1e-6)
            for i, value in enumerate(values):
                px = 40 + (width - 50) * i / n
                py = bottom - (float(value) - vmin) / span * (bottom - top)
                pts.extend([px, py])
            return pts

        x_pts = make_points(xs, -2.4, 2.4, 28, height - 16)
        cos_pts = make_points(coss, -1.0, 1.0, 28, height - 16)
        ret_max = max(max(rets), 1.0)
        ret_pts = make_points(rets, 0.0, ret_max, 28, height - 16)
        self.history_canvas.create_line(*x_pts, fill='#4c78a8', width=2)
        self.history_canvas.create_line(*cos_pts, fill='#54a24b', width=2)
        self.history_canvas.create_line(*ret_pts, fill='#f58518', width=2)

    def _draw_action_bars(self):
        self.bar_canvas.delete('all')
        trace = self.snapshot.get('last_trace') or {}
        candidates = (trace.get('candidates') or [])[:5]
        width = int(self.bar_canvas.winfo_width() or 420)
        height = int(self.bar_canvas.winfo_height() or 150)
        self.bar_canvas.create_text(10, 8, anchor='nw', text='Top actions', fill='#222222')
        if not candidates:
            self.bar_canvas.create_text(width / 2, height / 2, text='No trace yet', fill='#666666')
            return
        metric_key = 'q' if trace.get('mode') == 'sd_mcts' else 'score'
        metric_label = 'Q' if metric_key == 'q' else 'score'
        values = [float(row.get(metric_key, 0.0)) for row in candidates]
        vmax = max(abs(v) for v in values) or 1.0
        zero_y = height - 26
        left_margin = 28
        bar_gap = 14
        bar_w = max(18, int((width - left_margin * 2 - bar_gap * (len(candidates) - 1)) / max(1, len(candidates))))
        usable_h = height - 48
        self.bar_canvas.create_line(left_margin - 8, zero_y, width - 12, zero_y, fill='#999999')
        for idx, row in enumerate(candidates):
            value = float(row.get(metric_key, 0.0))
            x0 = left_margin + idx * (bar_w + bar_gap)
            x1 = x0 + bar_w
            bar_h = usable_h * abs(value) / vmax
            y0 = zero_y - bar_h
            fill = '#4c78a8' if idx > 0 else '#e45756'
            self.bar_canvas.create_rectangle(x0, y0, x1, zero_y, fill=fill, outline='')
            self.bar_canvas.create_text((x0 + x1) / 2, zero_y + 10, text=f"{row['action']:+.1f}", anchor='n', fill='#222222')
            self.bar_canvas.create_text((x0 + x1) / 2, y0 - 2, text=f"{value:+.2f}", anchor='s', fill='#222222')
        self.bar_canvas.create_text(width - 12, 8, anchor='ne', text=f'metric={metric_label}', fill='#666666')

    def _update_texts(self):
        state = self.snapshot['state']
        params = self.snapshot['params']
        summary = self.snapshot['summary']
        self.header_var.set(
            f"backend={self.snapshot.get('backend', 'unknown')}   env_step={self.snapshot['env_step']}   decision={self.snapshot['decision_count']}   return={self.snapshot['episode_return']:.1f}"
        )
        self.state_text.set(
            f"State\n"
            f"x = {state[0]:+.3f}\n"
            f"theta = {state[1]:+.3f}\n"
            f"x_dot = {state[2]:+.3f}\n"
            f"theta_dot = {state[3]:+.3f}\n"
            f"cos(theta) = {math.cos(float(state[1])):+.3f}"
        )
        initial = self.snapshot['initial_params']
        self.params_text.set(
            f"World model\n"
            f"g = {params['gravity']:.4f}  (Δ {params['gravity'] - initial['gravity']:+.4f})\n"
            f"mc = {params['masscart']:.4f}  (Δ {params['masscart'] - initial['masscart']:+.4f})\n"
            f"mp = {params['masspole']:.4f}  (Δ {params['masspole'] - initial['masspole']:+.4f})\n"
            f"l = {params['length']:.4f}  (Δ {params['length'] - initial['length']:+.4f})"
        )
        self.summary_text.set(
            f"Paper run summary\n"
            f"mean = {summary.get('mean_return', float('nan')):.1f}\n"
            f"std = {summary.get('std_return', float('nan')):.1f}\n"
            f"val_loss = {summary.get('val_loss', float('nan')):.4f}\n"
            f"mp_err = {summary.get('mp_err_percent', float('nan')):.2f}%"
        )

        cfg = self.snapshot['selected_cfg']
        self.selected_cfg_text.set(
            f"selector_mode = {cfg.get('selector_mode', '-') }\n"
            f"weight_decay = {cfg.get('weight_decay', '-') }\n"
            f"state_noise = {cfg.get('state_noise', '-') }\n"
            f"selection_metric = {cfg.get('selection_metric', '-') }\n"
            f"prior_tag = {cfg.get('prior_tag', '-') }"
        )

        self.trace_text.set(self._format_trace(self.snapshot['last_trace']))

    def _format_trace(self, trace):
        if not trace:
            return 'No action trace yet. Press Step or enable Auto-run.'
        lines = [
            f"mode = {trace.get('mode', '-')}",
            f"chosen_action = {trace.get('chosen_action', 0.0):+.3f}",
            f"current_vnet = {trace.get('current_vnet', 0.0):+.3f}",
        ]
        if 'num_simulations' in trace:
            lines.append(f"num_simulations = {trace['num_simulations']}")
        leaf = trace.get('leaf')
        if leaf:
            lines.extend([
                f"leaf_value = {leaf.get('leaf_value', 0.0):+.3f}",
                f"leaf_vnet = {leaf.get('vnet_value', 0.0):+.3f}",
                f"leaf_rollout = {leaf.get('rollout_value', 0.0) if leaf.get('rollout_value') is not None else float('nan'):+.3f}",
            ])
        lines.append('top actions:')
        candidates = trace.get('candidates', [])[:5]
        if not candidates:
            lines.append('  (none)')
        for row in candidates:
            if trace.get('mode') == 'sd_mcts':
                lines.append(
                    f"  a={row['action']:+.2f}  N={row.get('visits', 0):>3d}  Q={row.get('q', 0.0):+.3f}  edge={row.get('edge_reward', 0.0):+.3f}  cos'={row.get('next_cos_theta', 0.0):+.3f}"
                )
            elif trace.get('mode') == 'heuristic':
                lines.append(
                    f"  heuristic a={row['action']:+.2f}  score={row.get('score', 0.0):+.3f}  rollout={row.get('rollout_return', 0.0):+.3f}  edge={row.get('edge_reward', 0.0):+.3f}  cos'={row.get('next_cos_theta', 0.0):+.3f}"
                )
            else:
                lines.append(
                    f"  a={row['action']:+.2f}  score={row.get('score', 0.0):+.3f}  edge={row.get('edge_reward', 0.0):+.3f}  V'={row.get('next_vnet', 0.0):+.3f}  cos'={row.get('next_cos_theta', 0.0):+.3f}"
                )
        return '\n'.join(lines)

    def _on_close(self):
        self.running = False
        self.runtime._close_env()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='Cartpole SD-MCTS desktop demo')
    parser.add_argument('--smoke', action='store_true', help='Run headless smoke test instead of launching UI')
    parser.add_argument('--seed', type=int, default=42, help='Checkpoint seed for smoke test')
    parser.add_argument('--decisions', type=int, default=2, help='Decisions per mode in smoke test')
    args = parser.parse_args()

    if args.smoke:
        snapshots = run_headless_smoke(seed=args.seed, decisions=args.decisions)
        for idx, snap in enumerate(snapshots):
            trace_mode = snap['last_trace'].get('mode') if snap.get('last_trace') else None
            print(
                f"snap={idx} mode={snap['mode']} env_step={snap['env_step']} "
                f"return={snap['episode_return']:.3f} action={snap['last_action']:+.3f} trace={trace_mode}"
            )
        return

    root = tk.Tk()
    CartpoleDemoApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
