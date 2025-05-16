import streamlit as st

import base64
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename

import streamlit as st

import matplotlib.image as mpimg

import streamlit as st
import base64

import pandas as pd
import sqlite3

# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('2.avif')


def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path


if navigation() == "home":
    st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:50px;font-family:Caveat, sans-serif;">{"Amazon alexa review analysis"}</h1>', unsafe_allow_html=True)

    # st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:36px;">{"Ethical issues of Generative AI"}</h1>', unsafe_allow_html=True)
    
    print()
    print()

    print()

    st.text("                 ")
    st.text("                 ")
    a = " Alexa sentiment analysis of social media content has become increasingly significant in understanding public opinion and behavior."

    
    st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:24px;font-family:Caveat, sans-serif;">{a}</h1>', unsafe_allow_html=True)

    st.text("                 ")
    st.text("                 ")
    
    st.text("                 ")
    st.text("                 ")
    


elif navigation()=='reg':
   
    st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:50px;font-family:Caveat, sans-serif;">{"Amazon alexa review analysis"}</h1>', unsafe_allow_html=True)

    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:20px;">{"Register Here !!!"}</h1>', unsafe_allow_html=True)
    
    import streamlit as st
    import sqlite3
    import re
    
    # Function to create a database connection
    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn
    
    # Function to create a new user
    def create_user(conn, user):
        sql = ''' INSERT INTO users(name, password, email, phone)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        return cur.lastrowid
    
    # Function to check if a user already exists
    def user_exists(conn, email):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        if cur.fetchone():
            return True
        return False
    
    # Function to validate email
    def validate_email(email):
        pattern = r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
        return re.match(pattern, email)
    
    # Function to validate phone number
    def validate_phone(phone):
        pattern = r'^[6-9]\d{9}$'
        return re.match(pattern, phone)
    
    # Main function
    def main():
        # st.title("User Registration")
    
        # Create a database connection
        conn = create_connection("dbs.db")
    
        if conn is not None:
            # Create users table if it doesn't exist
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY,
                         name TEXT NOT NULL,
                         password TEXT NOT NULL,
                         email TEXT NOT NULL UNIQUE,
                         phone TEXT NOT NULL);''')
    
            # User input fields
            name = st.text_input("Enter your name")
            password = st.text_input("Enter your password", type="password")
            confirm_password = st.text_input("Confirm your password", type="password")
            email = st.text_input("Enter your email")
            phone = st.text_input("Enter your phone number")
    
            col1, col2 = st.columns(2)

            with col1:
                    
                aa = st.button("REGISTER")
                
                if aa:
                    
                    if password == confirm_password:
                        if not user_exists(conn, email):
                            if validate_email(email) and validate_phone(phone):
                                user = (name, password, email, phone)
                                create_user(conn, user)
                                st.success("User registered successfully!")
                            else:
                                st.error("Invalid email or phone number!")
                        else:
                            st.error("User with this email already exists!")
                    else:
                        st.error("Passwords do not match!")
                    
                    conn.close()
                    # st.success('Successfully Registered !!!')
                # else:
                    
                    # st.write('Registeration Failed !!!')     
            
            with col2:
                    
                aa = st.button("LOGIN")
                
                if aa:
                    import subprocess
                    subprocess.run(['python','-m','streamlit','run','login.py'])
                    # st.success('Successfully Registered !!!')
    
    
    
  
    if __name__ == '__main__':
        main()



elif navigation()=='Login':
    
    st.markdown(f'<h1 style="color:#000000 ;text-align: center;font-size:50px;font-family:Caveat, sans-serif;">{"Amazon alexa review analysis"}</h1>', unsafe_allow_html=True)


    # Function to create a database connection
    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print(e)
        return conn
    
    # Function to create a new user
    def create_user(conn, user):
        sql = ''' INSERT INTO users(name, password, email, phone)
                  VALUES(?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        return cur.lastrowid
    
    # Function to validate user credentials
    def validate_user(conn, name, password):
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE name=? AND password=?", (name, password))
        user = cur.fetchone()
        if user:
            return True, user[1]  # Return True and user name
        return False, None
    
    # Main function
    def main():
        # st.title("User Login")
        st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Login here"}</h1>', unsafe_allow_html=True)
    
    
        # Create a database connection
        conn = create_connection("dbs.db")
    
        if conn is not None:
            # Create users table if it doesn't exist
            conn.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY,
                         name TEXT NOT NULL,
                         password TEXT NOT NULL,
                         email TEXT NOT NULL UNIQUE,
                         phone TEXT NOT NULL);''')
    
            st.write("Enter your credentials to login:")
            name = st.text_input("User name")
            password = st.text_input("Password", type="password")
    
            col1, col2 = st.columns(2)
    
            with col1:
                    
                aa = st.button("Login")
                
                if aa:
    
    
            # # if st.button("Login"):
                    is_valid, user_name = validate_user(conn, name, password)
                    if is_valid:
                        st.success(f"Welcome back, {user_name}! Login successful!")
                        
                        import subprocess
                        subprocess.run(['python','-m','streamlit','run','Sentiment.py'])
                        
                        
                    else:
                        st.error("Invalid user name or password!")
                        
           
                      # st.success('Successfully Registered !!!')          
                        
                        
            # if st.button("Change Password"):
            #     import subprocess
            #     subprocess.run(['streamlit','run','CP.py'])
    
    
            # Close the database connection
            conn.close()
        else:
            st.error("Error! cannot create the database connection.")
    
    if __name__ == '__main__':
        main()



