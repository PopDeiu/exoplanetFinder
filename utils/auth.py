import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from utils.database import verify_credentials, register_user, get_user_by_id
import inspect
import os


def init_auth():
    caller_frame = inspect.stack()[1]
    caller_page = os.path.splitext(os.path.basename(caller_frame.filename))[0]
    st.session_state["_last_page"] = caller_page
    if 'auth_cookies' not in st.session_state:
        cookies = EncryptedCookieManager(
            password=os.getenv("COOKIE_PASSWORD", "parola-secreta-exoplanet-2026"),
        )
        if not cookies.ready():
            st.stop()
        st.session_state.auth_cookies = cookies

    if 'logged_in' not in st.session_state:
        cookies = st.session_state.auth_cookies
        saved_user_id = cookies.get('user_id')
        if saved_user_id:
            user_data = get_user_by_id(saved_user_id)
            if user_data:
                st.session_state.logged_in = True
                st.session_state.user_info = user_data
            else:
                st.session_state.logged_in = False
        else:
            st.session_state.logged_in = False


def render_sidebar_auth():
    cookies = st.session_state.get('auth_cookies')
    if not cookies:
        return

    if not st.session_state.logged_in:
        menu = st.radio("Navigare Cont", ["Login", "Înregistrare"], horizontal=True)

        if menu == "Login":
            with st.form("login_form"):
                st.subheader("Autentificare")
                user_in = st.text_input("Username")
                pass_in = st.text_input("Password", type="password")
                remember_me = st.checkbox("Ține-mă minte (Rămâi logat)")

                if st.form_submit_button("Log In", use_container_width=True):
                    user_data = verify_credentials(user_in, pass_in)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_data

                        if remember_me:
                            cookies['user_id'] = str(user_data['ID'])
                            cookies.save()

                        st.success(f"Salut, {user_in}!")
                        st.rerun()
                    else:
                        st.error("Credentiale incorecte")

                st.markdown("""
<style>
.forgot-msg { display: none; color: #e0e0ff; font-size: 0.85em; margin-top: 4px; padding: 6px; background: rgba(255,255,255,0.06); border-radius: 4px; }
#forgot:target { display: block; }
</style>
<a href="#forgot" style="color:#4da6ff; text-decoration:underline; font-size:0.85em; cursor:pointer;">Am uitat parola</a>
<div id="forgot" class="forgot-msg">Contactează administratorul la poprazvan09@gmail.com pentru resetarea parolei.</div>
""", unsafe_allow_html=True)

        else:
            with st.form("register_form"):
                st.subheader("Creare Cont Nou")
                new_user = st.text_input("Alege Username")
                new_pass = st.text_input("Alege Parolă", type="password")
                confirm_pass = st.text_input("Confirmă Parolă", type="password")

                if st.form_submit_button("Înregistrează-te", use_container_width=True):
                    if new_pass != confirm_pass:
                        st.error("Parolele nu coincid!")
                    elif len(new_user) < 3:
                        st.error("Username-ul este prea scurt!")
                    else:
                        success, message = register_user(new_user, new_pass)
                        if success:
                            st.success(message)
                            st.info("Acum te poți loga din meniul de Login.")
                        else:
                            st.error(message)
    else:
        username_display = st.session_state.user_info.get('user', 'Utilizator')
        st.write(f"✅ Logat ca: **{username_display}**")
        if st.button("Deconectare", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            if 'user_id' in cookies:
                del cookies['user_id']
                cookies.save()
            st.rerun()
