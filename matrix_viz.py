import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

st.set_page_config(
    page_title="Matrix Algo Viz",
    layout="wide",
    page_icon="🧩",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    h1 { color: #111827; font-weight: 800; letter-spacing: -1px;}
    h2 { color: #374151; font-weight: 600; }
    h3 { color: #4b5563; font-size: 1.1rem !important; }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1rem;
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        color: #2563eb;
    }
    .katex { font-size: 1.2em; color: #1e3a8a; }
</style>
""", unsafe_allow_html=True)

def standard_mult_bench(A, B):
    n = len(A)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def split(matrix):
    row, col = matrix.shape
    row2, col2 = row // 2, col // 2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]

def block_mult_bench(A, B):
    n = len(A)
    if n <= 2: return standard_mult_bench(A, B)
    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)
    C11 = block_mult_bench(A11, B11) + block_mult_bench(A12, B21)
    C12 = block_mult_bench(A11, B12) + block_mult_bench(A12, B22)
    C21 = block_mult_bench(A21, B11) + block_mult_bench(A22, B21)
    C22 = block_mult_bench(A21, B12) + block_mult_bench(A22, B22)
    return np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

def strassen_mult_bench(A, B):
    n = len(A)
    if n <= 2: return standard_mult_bench(A, B)
    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)
    m1 = strassen_mult_bench(A11 + A22, B11 + B22)
    m2 = strassen_mult_bench(A21 + A22, B11)
    m3 = strassen_mult_bench(A11, B12 - B22)
    m4 = strassen_mult_bench(A22, B21 - B11)
    m5 = strassen_mult_bench(A11 + A12, B22)
    m6 = strassen_mult_bench(A21 - A11, B11 + B12)
    m7 = strassen_mult_bench(A12 - A22, B21 + B22)
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    return np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

class Frame:
    def __init__(self, C, hl_A, hl_B, hl_C, desc, math):
        self.C = C.copy()
        self.hl_A = hl_A
        self.hl_B = hl_B
        self.hl_C = hl_C
        self.desc = desc
        self.math = math

@st.cache_data
def generate_frames_cached(algo_type, A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    frames = []

    if algo_type == "Standard":
        frames.append(Frame(C, None, None, None, "Start", "C = 0"))
        for i in range(n):
            for j in range(n):
                frames.append(Frame(C, (i, i + 1, 0, n), (0, n, j, j + 1), (i, i + 1, j, j + 1),
                                    f"Calculăm C[{i},{j}]", f"C_{{{i}{j}}} = Linia_{{A}} \\cdot Coloana_{{B}}"))
                val = 0
                for k in range(n):
                    val += A[i, k] * B[k, j]
                    C[i, j] = val
                    frames.append(Frame(C, (i, i + 1, k, k + 1), (k, k + 1, j, j + 1), (i, i + 1, j, j + 1),
                                        f"Produs elementar: {A[i, k]} * {B[k, j]}",
                                        f"C_{{{i}{j}}} \\leftarrow {val - A[i, k] * B[k, j]:.0f} + {A[i, k]} \\cdot {B[k, j]}"))
        frames.append(Frame(C, None, None, None, "Finalizat", "C = A \\times B"))

    elif algo_type == "Block":
        frames.append(Frame(C, None, None, None, "Inițializare", "Divide et Impera"))
        mid = n // 2
        if n < 2:
            return generate_frames_cached("Standard", A, B)

        quads = [
            ('C11', (0, mid, 0, mid), [((0, mid, 0, mid), (0, mid, 0, mid)), ((0, mid, mid, n), (mid, n, 0, mid))]),
            ('C12', (0, mid, mid, n), [((0, mid, 0, mid), (0, mid, mid, n)), ((0, mid, mid, n), (mid, n, mid, n))]),
            ('C21', (mid, n, 0, mid), [((mid, n, 0, mid), (0, mid, 0, mid)), ((mid, n, mid, n), (mid, n, 0, mid))]),
            ('C22', (mid, n, mid, n), [((mid, n, 0, mid), (0, mid, mid, n)), ((mid, n, mid, n), (mid, n, mid, n))])
        ]

        for name, (rs, re, cs, ce), ops in quads:
            a1, b1 = ops[0]
            frames.append(Frame(C, a1, b1, (rs, re, cs, ce), f"{name}: Calculăm primul termen",
                                f"{name} = A_{{b1}} \\cdot B_{{b1}}"))
            C[rs:re, cs:ce] += A[a1[0]:a1[1], a1[2]:a1[3]] @ B[b1[0]:b1[1], b1[2]:b1[3]]

            a2, b2 = ops[1]
            frames.append(Frame(C, a2, b2, (rs, re, cs, ce), f"{name}: Adăugăm al doilea termen",
                                f"{name} += A_{{b2}} \\cdot B_{{b2}}"))
            C[rs:re, cs:ce] += A[a2[0]:a2[1], a2[2]:a2[3]] @ B[b2[0]:b2[1], b2[2]:b2[3]]

    elif algo_type == "Strassen":
        frames.append(Frame(C, None, None, None, "Pregătire Strassen", "Descompunere în 7 matrici M"))
        if n < 2: return generate_frames_cached("Standard", A, B)
        mid = n // 2
        steps = [
            ("M1", (0, n, 0, n), (0, n, 0, n), "M_1 = (A_{11}+A_{22})(B_{11}+B_{22})"),
            ("M2", (mid, n, 0, n), (0, mid, 0, mid), "M_2 = (A_{21}+A_{22})B_{11}"),
            ("M3", (0, mid, 0, mid), (0, n, mid, n), "M_3 = A_{11}(B_{12}-B_{22})"),
            ("M4", (mid, n, mid, n), (0, n, 0, mid), "M_4 = A_{22}(B_{21}-B_{11})"),
            ("M5", (0, mid, 0, n), (mid, n, mid, n), "M_5 = (A_{11}+A_{12})B_{22}"),
            ("M6", (0, n, 0, mid), (0, mid, 0, n), "M_6 = (A_{21}-A_{11})(B_{11}+B_{12})"),
            ("M7", (0, n, mid, n), (mid, n, 0, n), "M_7 = (A_{12}-A_{22})(B_{21}+B_{22})")
        ]
        for name, hA, hB, form in steps:
            frames.append(Frame(C, hA, hB, None, f"Calcul {name}", f"$${form}$$"))

        C = A @ B
        frames.append(Frame(C, None, None, (0, n, 0, n), "Combinare & Rezultat", "C_f = Combinare liniară(M_1...M_7)"))

    return frames

def plot_styled(A, B, fr):
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor='none')
    fig.subplots_adjust(wspace=0.1)

    cmaps = ['Purples', 'Oranges', 'GnBu']
    titles = ["Matricea A", "Matricea B", "Rezultat C"]
    matrices = [A, B, fr.C]
    highlights = [fr.hl_A, fr.hl_B, fr.hl_C]

    for i, ax in enumerate(axes):
        mat = matrices[i]
        hl = highlights[i]
        im = ax.imshow(mat, cmap=cmaps[i], vmin=0, vmax=np.max(mat) + 1, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i], fontsize=14, pad=10, fontweight='bold', color='#374151')

        rows, cols = mat.shape
        for r in range(rows + 1):
            ax.axhline(r - 0.5, color='grey', lw=0.5, alpha=0.2)
        for c in range(cols + 1):
            ax.axvline(c - 0.5, color='grey', lw=0.5, alpha=0.2)

        for r in range(rows):
            for c in range(cols):
                active = False
                if hl:
                    rs, re, cs, ce = hl
                    if rs <= r < re and cs <= c < ce: active = True
                val_str = f"{int(mat[r, c])}"
                if active:
                    rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, fill=False, edgecolor='#ef4444', lw=2.5, zorder=10)
                    ax.add_patch(rect)
                    ax.text(c, r, val_str, ha='center', va='center', color='black', fontweight='bold', fontsize=14, zorder=11)
                else:
                    ax.text(c, r, val_str, ha='center', va='center', color='#4b5563', fontsize=11)
    return fig

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620580.png", width=50)
    st.title("Vizualizare Matrici")
    st.markdown("Control panel pentru simulare.")

    st.subheader("⚙️ Configurare")

    if 'viz_data' not in st.session_state:
        st.session_state.viz_data = (np.random.randint(1, 6, (4, 4)), np.random.randint(1, 6, (4, 4)))
    if 'viz_running' not in st.session_state: st.session_state.viz_running = False
    if 'step_idx' not in st.session_state: st.session_state.step_idx = 0

    size_sel = st.selectbox("Dimensiune Matrice (N)", [2, 4, 8], index=1)
    algo_sel = st.selectbox("Algoritm", ["Standard", "Block", "Strassen"])

    current_N = st.session_state.viz_data[0].shape[0]
    if size_sel != current_N:
        st.session_state.viz_data = (np.random.randint(1, 6, (size_sel, size_sel)), np.random.randint(1, 6, (size_sel, size_sel)))
        st.session_state.step_idx = 0
        st.session_state.viz_running = False
        st.rerun()

    speed_sel = st.slider("Viteză animație (sec/pas)", 1.0, 3.0, 1.5, 0.1)

    st.divider()

    c1, c2 = st.columns(2)
    start = c1.button("▶ START", use_container_width=True, type="primary")
    stop = c2.button("⏹ STOP", use_container_width=True)
    reset = st.button("⏮ RESET", use_container_width=True)

    if start: st.session_state.viz_running = True
    if stop: st.session_state.viz_running = False
    if reset:
        st.session_state.viz_running = False
        st.session_state.step_idx = 0

tab_viz, tab_bench = st.tabs(["🎨 Vizualizare Interactivă", "🚀 Benchmark & Performanță"])

with tab_viz:
    st.markdown(f"## 🖥️ Algoritmul: {algo_sel}")

    A_viz, B_viz = st.session_state.viz_data
    frames = generate_frames_cached(algo_sel, A_viz, B_viz)

    col_main, col_info = st.columns([3, 1])

    with col_main:
        curr_frame = frames[min(st.session_state.step_idx, len(frames) - 1)]
        fig = plot_styled(A_viz, B_viz, curr_frame)
        st.pyplot(fig, use_container_width=True)
        st.latex(curr_frame.math)

    with col_info:
        st.metric("Pasul Curent", f"{st.session_state.step_idx + 1} / {len(frames)}")
        st.markdown("---")
        st.info(f"**Acțiune:**\n\n{curr_frame.desc}")
        st.progress(min((st.session_state.step_idx + 1) / len(frames), 1.0))

        with st.expander("ℹ️ Detalii Teoretice", expanded=True):
            if algo_sel == "Standard":
                st.write("Metoda clasică. Iterează linie cu coloană.")
                st.code("Complexitate: O(N³)")
            elif algo_sel == "Block":
                st.write("Divide matricea în 4 blocuri mai mici recursiv. Bun pentru cache.")
                st.code("O(N³) optimizat")
            else:
                st.write("Reduce nr. de înmulțiri de la 8 la 7 per recursie.")
                st.code("Complexitate: O(N^2.81)")

    if st.session_state.viz_running:
        if st.session_state.step_idx < len(frames) - 1:
            time.sleep(speed_sel)
            st.session_state.step_idx += 1
            st.rerun()
        else:
            st.session_state.viz_running = False
            st.balloons()
            st.success("Simulare completă!")

with tab_bench:
    st.markdown("### 🏎️ Cursă de viteză între algoritmi")
    st.info("Comparăm Numpy (backend C/Fortran) cu implementările pure Python.")

    col_b_input, col_b_res = st.columns([1, 2])

    with col_b_input:
        bench_n = st.selectbox("Mărime Matrice (N)", [32, 64, 128, 256, 512], index=2)
        run_bench = st.button("🔥 Rulează Benchmark", use_container_width=True, type="primary")

    if run_bench:
        with col_b_res:
            with st.spinner(f"Calculăm pentru matrici {bench_n}x{bench_n}..."):
                A_b = np.random.rand(bench_n, bench_n)
                B_b = np.random.rand(bench_n, bench_n)
                res = []

                t0 = time.time()
                np.dot(A_b, B_b)
                t_np = max(time.time() - t0, 0.000001)
                res.append({"Algo": "Numpy (Optimizat)", "Timp (s)": t_np, "Color": "#10b981"})

                if bench_n > 64:
                    res.append({"Algo": "Standard (Python)", "Timp (s)": 0, "Color": "#ef4444"})
                    skip_std = True
                else:
                    t0 = time.time()
                    standard_mult_bench(A_b, B_b)
                    t_std = max(time.time() - t0, 0.000001)
                    res.append({"Algo": "Standard (Python)", "Timp (s)": t_std, "Color": "#ef4444"})
                    skip_std = False

                try:
                    t0 = time.time()
                    block_mult_bench(A_b, B_b)
                    t_blk = max(time.time() - t0, 0.000001)
                    res.append({"Algo": "Block (Recursiv)", "Timp (s)": t_blk, "Color": "#f59e0b"})
                except:
                    pass

                df = pd.DataFrame(res)
                valid_df = df[df["Timp (s)"] > 0].sort_values("Timp (s)")

                st.bar_chart(valid_df, x="Algo", y="Timp (s)", color="Color")
                st.table(df[["Algo", "Timp (s)"]])

                if not skip_std and t_std > 0:
                    speedup = t_std / t_np
                    st.success(f"Numpy a fost de **{speedup:.1f}x** mai rapid decât codul Python standard!")
                elif skip_std:
                    st.warning("Standard Python a fost sărit (prea lent pentru N > 64).")
