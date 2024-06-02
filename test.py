import firebase_admin
import streamlit as st
from firebase_admin import auth
from firebase_admin import credentials

if not firebase_admin._apps:
    cred = credentials.Certificate("analysis-2ee6b-109d7b5595b8.json")
    firebase_admin.initialize_app(cred)


def app():
    st.title('Welcome to Whatsapp Chat Analyzer')

    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''

    def f():
        try:
            user = auth.get_user_by_email(email)
            print(user.uid)
            st.session_state.username = user.uid
            st.session_state.useremail = user.email

            # Redirect to chat analysis app with the username as a query parameter
            st.experimental_set_query_params(username=user.uid)

            st.session_state.signedout = True
            st.session_state.signout = True
        except:
            st.warning('Login Failed')

    def t():
        st.session_state.signout = False
        st.session_state.signedout = False
        st.session_state.username = ''

    if "signedout" not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False

    if not st.session_state["signedout"]:  # only show if the state is False, hence the button has never been clicked
        choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')

        if choice == 'Sign up':
            username = st.text_input("Enter  your unique username")

            if st.button('Create my account'):
                user = auth.create_user(email=email, password=password, uid=username)

                st.success('Account created successfully!')
                st.markdown('Please Login using your email and password')
                st.balloons()
        else:
            # st.button('Login', on_click=f)
            st.button('Login', on_click=f)

    if st.session_state.signout:
        st.text('Name ' + st.session_state.username)
        st.text('Email id: ' + st.session_state.useremail)
        st.button('Sign out', on_click=t)


