"""
make_table.py: monta a tabela de auditoria (seção 13 do protocolo) a partir de
todos os JSONs em runs/ e dos audit_run*.json, em markdown e CSV.

    python make_table.py                    # lê runs/*.json e audit_*.json na pasta atual
    python make_table.py --out tabela.md
"""
import argparse, csv, glob, json, math, os

PI = math.pi
MIN = {0: 4 * PI, 1: 2 * PI ** 2}


def row(a):
    g = a.get("genus")
    W = a.get("W")
    est = a.get("estimator") or {}
    rs = a.get("resolution_study") or {}
    c1, c2, c3, c4 = a.get("C1") or {}, a.get("C2") or {}, a.get("C3") or {}, a.get("C4") or {}
    var = a.get("C4_variational") or {}
    grid = a.get("grid"); grid = grid[0] if isinstance(grid, (list, tuple)) else grid
    return dict(
        surface=a.get("surface") or a.get("label") or a.get("ckpt", "?"),
        genus=g,
        grid=f"{grid}²" + ("" if rs.get("converged", True) else " (not converged)"),
        W_cert=f"{W:.4f}" if W is not None else "",
        ratio=f"{W / MIN[g]:.4f}" if (W is not None and g in MIN) else "",
        W_rep=f"{est['W_rep_mean']:.3f} ± {est['W_rep_sd']:.3f}" if est else "",
        bias=f"{est['bias']:+.3f}{'*' if est.get('bias_significant') else ''}" if est else "",
        huber="active" if est.get("huber_active") else ("no" if est else ""),
        saved_loss=f"{a['ckpt_loss']:.4f}" if a.get("ckpt_loss") is not None else (f"{a['meta']['reported_loss']:.4f}" if a.get("meta", {}).get("reported_loss") is not None else ""),
        C1=f"χ̂={c1.get('chi_estimate', float('nan')):.3f}, {c1.get('residual', float('nan')):.1e}" if c1 else "",
        C2=f"{c2.get('rel_diff_max', c2.get('rel_diff', float('nan'))):.1e}" if c2 else "",
        C3=("embedded" if c3.get("embedded_certified") else f"mult ≤ {c3.get('max_multiplicity_bound')}") if c3 else "",
        C4=f"{c4.get('residual_relative', float('nan')):.2f} {c4.get('band', '')}" if c4 else "",
        C4_fallback=f"{var['residual_L2_lower_bound']:.3f}" if var else "",
        verdicts=", ".join(k.split()[0] for k, v in (a.get("verdicts") or {}).items() if not v) or "all pass",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit_table.md")
    a = ap.parse_args()
    files = sorted(glob.glob("runs/audit__*.json")) + sorted(glob.glob("audit_*.json"))
    rows = []
    for p in files:
        try:
            with open(p) as f:
                d = json.load(f)
            if d.get("kind", "audit") != "audit":
                continue
            r = row(d); r["file"] = os.path.basename(p); rows.append(r)
        except Exception as e:
            print("skip", p, e)
    if not rows:
        print("nenhum JSON de auditoria encontrado"); return
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    open(a.out, "w").write("\n".join(md) + "\n")
    with open(a.out.replace(".md", ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("\n".join(md)); print(f"\nsalvo em {a.out} e {a.out.replace('.md', '.csv')}   (* = viés significativo)")


if __name__ == "__main__":
    main()
