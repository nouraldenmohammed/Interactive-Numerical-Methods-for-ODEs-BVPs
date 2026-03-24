# pip install streamlit
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# NUMERICAL METHODS (Generalized for scalars and systems)
# --------------------------------------------------------
def euler_method(f, a, b, n, y0):
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    if np.ndim(y0) == 0:
        y = np.zeros(n + 1)
    else:
        y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(t[i], y[i])
    return t, y

def rk2_method(f, a, b, n, y0):
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    if np.ndim(y0) == 0:
        y = np.zeros(n + 1)
    else:
        y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i+1], y[i] + k1)
        y[i+1] = y[i] + 0.5 * (k1 + k2)
    return t, y

def midpoint_method(f, a, b, n, y0):
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    if np.ndim(y0) == 0:
        y = np.zeros(n + 1)
    else:
        y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        y[i+1] = y[i] + k2
    return t, y

def rk4_method(f, a, b, n, y0):
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    if np.ndim(y0) == 0:
        y = np.zeros(n + 1)
    else:
        y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        k3 = h * f(t[i] + h/2, y[i] + k2/2)
        k4 = h * f(t[i+1], y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, y

def bvp_finite_difference(p, q, r, a, b, alpha, beta, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    A = np.zeros((n - 1, n - 1))
    d = np.zeros(n - 1)
    
    for i in range(1, n):
        xi = x[i]
        a_coeff = 1 - (h/2) * p(xi)
        b_coeff = -2 + (h**2) * q(xi)
        c_coeff = 1 + (h/2) * p(xi)
        
        idx = i - 1
        A[idx, idx] = b_coeff
        
        if idx > 0:
            A[idx, idx - 1] = a_coeff
        if idx < n - 2:
            A[idx, idx + 1] = c_coeff
            
        d[idx] = (h**2) * r(xi)
        if i == 1:
            d[idx] -= a_coeff * alpha
        if i == n - 1:
            d[idx] -= c_coeff * beta
            
    y_inner = np.linalg.solve(A, d)
    y = np.zeros(n + 1)
    y[0] = alpha
    y[n] = beta
    y[1:n] = y_inner
    
    return x, y

# --------------------------------------------------------
# HELPER FOR MATH EVALUATION
# --------------------------------------------------------
def get_eval_env(**kwargs):
    """Creates a safe evaluation environment with common math functions."""
    env = {"np": np, "exp": np.exp, "sin": np.sin, "cos": np.cos,
           "tan": np.tan, "log": np.log, "sqrt": np.sqrt, "pi": np.pi, "e": np.e}
    env.update(kwargs)
    return env

# --------------------------------------------------------
# STREAMLIT APP LAYOUT
# --------------------------------------------------------
st.set_page_config(page_title="Numerical ODEs Applet", layout="wide")
st.title("Interactive Numerical Methods for ODEs & BVPs")
st.markdown("*By Dr Nouralden Mohammed*")

tab1, tab2, tab3 = st.tabs(["1st Order ODEs", "Systems (2nd Order)", "Boundary Value Problems"])

# --- TAB 1: 1st ORDER ODEs ---
with tab1:
    st.header("Customizable 1st Order ODE")
    st.markdown("Define your ODE of the form $y' = f(t, y)$. Standard functions like `exp()`, `sin()`, and `cos()` are supported.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**ODE Definition:**")
        ode_expr = st.text_input("f(t, y):", value="y * (1 - y)")
        exact_expr = st.text_input("Exact y(t) (optional):", value="(y0 * exp(t)) / (1 - y0 + y0 * exp(t))")
        
        st.markdown("**Parameters:**")
        c1, c2 = st.columns(2)
        with c1:
            t0_log = st.number_input("Start t0", value=0.0)
            y0_log = st.number_input("Initial y0", value=0.5)
        with c2:
            tf_log = st.number_input("End tf", value=5.0)
            n_log = st.slider("Steps (n)", 2, 200, 10)
            
        methods_selected = st.multiselect(
            "Select Numerical Methods:",
            ["Euler", "RK2 (Modified Euler)", "Midpoint", "RK4"],
            default=["Euler", "RK4"]
        )
    
    with col2:
        try:
            def f_custom(t, y):
                return eval(ode_expr, get_eval_env(t=t, y=y, y0=y0_log))
            
            t_exact = np.linspace(t0_log, tf_log, 200)
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            
            if exact_expr.strip():
                y_exact = eval(exact_expr, get_eval_env(t=t_exact, y0=y0_log))
                if np.isscalar(y_exact):
                    y_exact = np.ones_like(t_exact) * y_exact
                ax1.plot(t_exact, y_exact, 'k-', label='Exact Solution', linewidth=2)
            
            if "Euler" in methods_selected:
                t_num, y_num = euler_method(f_custom, t0_log, tf_log, n_log, y0_log)
                ax1.plot(t_num, y_num, 'o--', label='Euler')
            if "RK2 (Modified Euler)" in methods_selected:
                t_num, y_num = rk2_method(f_custom, t0_log, tf_log, n_log, y0_log)
                ax1.plot(t_num, y_num, 's--', label='RK2')
            if "Midpoint" in methods_selected:
                t_num, y_num = midpoint_method(f_custom, t0_log, tf_log, n_log, y0_log)
                ax1.plot(t_num, y_num, '^--', label='Midpoint')
            if "RK4" in methods_selected:
                t_num, y_num = rk4_method(f_custom, t0_log, tf_log, n_log, y0_log)
                ax1.plot(t_num, y_num, '*--', label='RK4')
                
            ax1.set_xlabel('Time (t)')
            ax1.set_ylabel('y(t)')
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
        except Exception as e:
            st.error(f"Error evaluating mathematical expression: {e}")

# --- TAB 2: SYSTEMS OF ODEs ---
with tab2:
    st.header("Customizable System of 2 ODEs")
    st.markdown("Define a system: $u' = f_u(t, u, v)$ and $v' = f_v(t, u, v)$")
    
    col3, col4 = st.columns([1, 3])
    with col3:
        st.markdown("**System Definition:**")
        u_expr = st.text_input("u' = f_u(t, u, v):", value="v")
        v_expr = st.text_input("v' = f_v(t, u, v):", value="-2*u - 3*v")
        exact_sys_expr = st.text_input("Exact u(t) (optional):", value="(2*u0 + v0)*exp(-t) + (-u0 - v0)*exp(-2*t)")
        
        st.markdown("**Parameters:**")
        c3, c4 = st.columns(2)
        with c3:
            t0_sys = st.number_input("Start t0 ", value=0.0)
            u0_sys = st.number_input("Initial u(0)", value=1.0)
        with c4:
            tf_sys = st.number_input("End tf ", value=2.0)
            v0_sys = st.number_input("Initial v(0)", value=0.0)
        n_sys = st.slider("Steps (n)", 2, 200, 20)
        sys_methods = st.multiselect(
            "Select Methods to Compare:",
            ["Euler", "RK4"],
            default=["Euler", "RK4"]
        )

    with col4:
        try:
            def f_sys_custom(t, y_vec):
                u, v = y_vec
                du = eval(u_expr, get_eval_env(t=t, u=u, v=v, u0=u0_sys, v0=v0_sys))
                dv = eval(v_expr, get_eval_env(t=t, u=u, v=v, u0=u0_sys, v0=v0_sys))
                return np.array([du, dv])
                
            t_ex_sys = np.linspace(t0_sys, tf_sys, 200)
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            
            if exact_sys_expr.strip():
                y_ex_sys = eval(exact_sys_expr, get_eval_env(t=t_ex_sys, u0=u0_sys, v0=v0_sys))
                if np.isscalar(y_ex_sys):
                    y_ex_sys = np.ones_like(t_ex_sys) * y_ex_sys
                ax2.plot(t_ex_sys, y_ex_sys, 'k-', label='Exact u(t)', linewidth=2)
            
            y0_sys = np.array([u0_sys, v0_sys])
            if "Euler" in sys_methods:
                t_sys, y_sys = euler_method(f_sys_custom, t0_sys, tf_sys, n_sys, y0_sys)
                ax2.plot(t_sys, y_sys[:, 0], 'o--', label='Euler $u(t)$')
            if "RK4" in sys_methods:
                t_sys, y_sys = rk4_method(f_sys_custom, t0_sys, tf_sys, n_sys, y0_sys)
                ax2.plot(t_sys, y_sys[:, 0], '*--', label='RK4 $u(t)$')
                
            ax2.set_xlabel('Time (t)')
            ax2.set_ylabel('u(t)')
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error evaluating mathematical expression: {e}")

# --- TAB 3: BOUNDARY VALUE PROBLEMS ---
with tab3:
    st.header("Customizable Linear BVP")
    st.markdown("Equation: $y'' + p(x)y' + q(x)y = r(x)$")
    
    col5, col6 = st.columns([1, 3])
    with col5:
        st.markdown("**Equation Components:**")
        p_expr = st.text_input("p(x):", value="x + 1")
        q_expr = st.text_input("q(x):", value="-2")
        r_expr = st.text_input("r(x):", value="(1 - x**2) * exp(-x)")
        exact_bvp_expr = st.text_input("Exact y(x) (optional):", value="(x - 1) * exp(-x)")
        
        st.markdown("**Boundaries & Grid:**")
        c5, c6 = st.columns(2)
        with c5:
            a_bvp = st.number_input("Start x (a)", value=0.0)
            alpha_bvp = st.number_input("y(a)", value=-1.0)
        with c6:
            b_bvp = st.number_input("End x (b)", value=1.0)
            beta_bvp = st.number_input("y(b)", value=0.0)
        n_bvp = st.slider("Grid Points (n)", 3, 100, 5)
        if n_bvp > 0:
            st.info(f"Step size $h = {(b_bvp - a_bvp)/n_bvp:.4f}$")
        
    with col6:
        try:
            def p_custom(x): return eval(p_expr, get_eval_env(x=x))
            def q_custom(x): return eval(q_expr, get_eval_env(x=x))
            def r_custom(x): return eval(r_expr, get_eval_env(x=x))
            
            x_ex_bvp = np.linspace(a_bvp, b_bvp, 200)
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            
            if exact_bvp_expr.strip():
                y_ex_bvp = eval(exact_bvp_expr, get_eval_env(x=x_ex_bvp))
                if np.isscalar(y_ex_bvp):
                    y_ex_bvp = np.ones_like(x_ex_bvp) * y_ex_bvp
                ax3.plot(x_ex_bvp, y_ex_bvp, 'k-', label='Exact solution', linewidth=2)
            
            x_bvp, y_bvp = bvp_finite_difference(p_custom, q_custom, r_custom, a_bvp, b_bvp, alpha_bvp, beta_bvp, n_bvp)
            
            ax3.plot(x_bvp, y_bvp, '*r--', label=f'Difference solution ($n={n_bvp}$)', markersize=8)
            ax3.set_xlabel('x')
            ax3.set_ylabel('y(x)')
            ax3.legend()
            ax3.grid(True)
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"Error evaluating mathematical expression: {e}")