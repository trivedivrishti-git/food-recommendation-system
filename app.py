import math
import pickle
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Food Recommender", layout="wide")
DEFAULT_IMG = "https://media.istockphoto.com/vectors/no-image-vector-symbol-missing-available-icon-no-gallery-for-this-vector-id1424817878?k=20&m=1424817878&s=612x612&w=0&h=aRDODu1B4F0rOqbbI9QaqSjBvfYTLsdDsEGswrImnoU="

st.markdown(
    """
<style>
    [data-testid="stImage"] img {
        height: 200px !important;
        object-fit: cover !important;
        border-radius: 10px !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 24px;
        border-bottom: 1px solid #e6e9ef;
        padding-bottom: 0px;
        margin-bottom: 1rem;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] label {
        padding: 8px 4px 10px 4px !important;
        margin: 0 !important;
        border-bottom: 2px solid transparent;
        cursor: pointer;
        background: transparent !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] label p {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #31333F;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
        border-bottom: 2px solid #ff4b4b !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) p {
        color: #ff4b4b !important;
        font-weight: 600 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)
st.title("FOOD RECOMMENDATION SYSTEM")

FLASK_API_URL = "http://127.0.0.1:5000/recommend"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "full_name" not in st.session_state:
    st.session_state.full_name = ""
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "HOME"
if "last_saved_dish" not in st.session_state:
    st.session_state.last_saved_dish = ""


def clean_and_normalize_df(df_to_clean):
    df_to_clean["name"] = df_to_clean["name"].fillna("").astype(str).str.strip()
    df_to_clean["diet"] = (
        df_to_clean["diet"].fillna("").astype(str).str.strip().str.title()
    )
    df_to_clean["course"] = (
        df_to_clean["course"].fillna("").astype(str).str.strip().str.title()
    )
    df_to_clean["ingredients"] = (
        df_to_clean["ingredients"].fillna("").astype(str).str.strip()
    )
    df_to_clean["img_url"] = (
        df_to_clean["img_url"].fillna("").astype(str).str.strip()
    )
    return df_to_clean


@st.cache_data
def load_data_from_file():
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    model_data["df"] = clean_and_normalize_df(model_data["df"])
    return model_data


try:
    model = load_data_from_file()
    df = model["df"]
except Exception:
    st.error("Please run 'python train.py' first!")
    st.stop()


def save_updated_df(updated_df):
    updated_df = clean_and_normalize_df(updated_df)

    updated_df["tags"] = (
            updated_df["ingredients"].str.lower()
            + " "
            + updated_df["diet"].str.lower()
            + " "
            + updated_df["course"].str.lower()
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(updated_df["tags"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    model_data = {
        "df": updated_df,
        "tfidf_vectorizer": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "similarity_matrix": cosine_sim,
        "cosine_sim": cosine_sim,
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    try:
        updated_df.to_csv("food_dataset.csv", index=False)
    except Exception as e:
        st.warning(f"CSV update error: {e}")

    st.cache_data.clear()
    st.cache_resource.clear()

    try:
        requests.get("http://127.0.0.1:5000/reload", timeout=2)
    except Exception:
        pass


conn = sqlite3.connect("food.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT,
        email TEXT,
        phone TEXT,
        username TEXT UNIQUE,
        password TEXT
    )
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    selected_food TEXT,
    diet TEXT,
    course TEXT,
    max_price REAL
)
""")
conn.commit()


def register_user(full_name, email, phone, username, password):
    try:
        cursor.execute(
            "INSERT INTO users(full_name, email, phone, username, password)"
            " VALUES (?,?,?,?,?)",
            (full_name, email, phone, username, password),
        )
        conn.commit()
        return True
    except Exception:
        return False


def login_user(username, password):
    return cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password),
    ).fetchone()


def save_history(username, food, diet, course, max_price):
    cursor.execute(
        "INSERT INTO history(username, selected_food, diet, course, max_price)"
        " VALUES (?,?,?,?,?)",
        (username, food, diet, course, max_price),
    )
    conn.commit()


def get_history(username):
    return pd.read_sql_query(
        """
        SELECT selected_food, MAX(id) as max_id 
        FROM history 
        WHERE username=? AND selected_food IS NOT NULL AND selected_food != '' 
        GROUP BY selected_food 
        ORDER BY max_id DESC
        """,
        conn,
        params=(username,),
    )


def clear_history(username):
    cursor.execute("DELETE FROM history WHERE username=?", (username,))
    conn.commit()


def remove_single_history(username, food_name):
    cursor.execute(
        "DELETE FROM history WHERE username=? AND selected_food=?",
        (username, food_name),
    )
    conn.commit()


def on_tab_change():
    st.session_state.current_tab = st.session_state.nav_radio


def set_course_callback(course_name):
    st.session_state.active_course = course_name
    st.session_state[f"dish_page_{course_name}"] = 1


def set_home_page_callback(page_key, page_num):
    st.session_state[page_key] = page_num


def set_history_page_callback(page_num):
    st.session_state.history_page = page_num


def handle_remove_single_callback(username, food_name):
    remove_single_history(username, food_name)


def handle_clear_all_callback(username):
    clear_history(username)
    st.session_state.history_page = 1


def handle_logout_callback():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.full_name = ""
    st.session_state.is_admin = False
    st.session_state.current_tab = "HOME"


def render_side_slider(dishes):
    if not dishes:
        return
    cards_html = "".join([
        f"""
            <div style="min-width:190px;max-width:190px;border:1px solid #e0e0e0;border-radius:10px;padding:10px;background:#fff;box-shadow:0 2px 5px rgba(0,0,0,0.08);flex-shrink:0;">
                <img src="{d.get('img_url') if d.get('img_url') else DEFAULT_IMG}" style="width:100%;height:130px;object-fit:cover;border-radius:8px;">
                <h4 style="margin:8px 0 4px;font-size:14px;font-family:sans-serif;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#111;">{d.get('name', 'Dish')}</h4>
                <p style="margin:2px 0;font-size:12px;font-family:sans-serif;color:#444;"><b>Price:</b> ₹{d.get('price', 0)}</p>
                <p style="margin:2px 0;font-size:12px;font-family:sans-serif;color:#444;"><b>Course:</b> {d.get('course', '')}</p>
            </div>
    """
        for d in dishes
    ])

    slider_code = f"""
    <!DOCTYPE html><html><head><style>
        body {{ margin: 0; padding: 0; background: transparent; font-family: sans-serif; }}
        .slider-box {{ position: relative; display: flex; align-items: center; width: 100%; padding: 5px 0; }}
        .cards-container {{ display: flex; overflow-x: auto; gap: 15px; padding: 10px 40px; scroll-behavior: smooth; scrollbar-width: none; }}
        .cards-container::-webkit-scrollbar {{ display: none; }}
        .nav-arrow {{ position: absolute; top: 50%; transform: translateY(-50%); z-index: 10; background: #fff; border: 1px solid #ccc; border-radius: 50%; width: 34px; height: 34px; cursor: pointer; font-size: 16px; font-weight: bold; box-shadow: 0 2px 6px rgba(0,0,0,0.2); display: flex; align-items: center; justify-content: center; color: #333; transition: opacity 0.3s, visibility 0.3s; }}
        .nav-arrow:hover {{ background: #f0f0f0; }}
        .arrow-left {{ left: 2px; }}
        .arrow-right {{ right: 2px; }}
        .hidden {{ opacity: 0; visibility: hidden; pointer-events: none; }}
    </style></head><body>
        <div class="slider-box">
            <button id="btn-left" class="nav-arrow arrow-left hidden" onclick="scrollContainer(-320)">❮</button>
            <div class="cards-container" id="track">{cards_html}</div>
            <button id="btn-right" class="nav-arrow arrow-right" onclick="scrollContainer(320)">❯</button>
        </div>
        <script>
            const track = document.getElementById('track'), btnL = document.getElementById('btn-left'), btnR = document.getElementById('btn-right');
            function checkNav() {{
                const max = track.scrollWidth - track.clientWidth;
                track.scrollLeft <= 5 ? btnL.classList.add('hidden') : btnL.classList.remove('hidden');
                track.scrollLeft >= max - 5 ? btnR.classList.add('hidden') : btnR.classList.remove('hidden');
            }}
            function scrollContainer(v) {{ track.scrollBy({{ left: v, behavior: 'smooth' }}); }}
            track.addEventListener('scroll', checkNav);
            window.addEventListener('load', checkNav);
            setTimeout(checkNav, 300);
        </script>
    </body></html>
    """
    components.html(slider_code, height=240)


if st.session_state.logged_in:
    st.sidebar.success(f"Welcome {st.session_state.full_name}")
    st.sidebar.header("Filter Recommendation")
    food_list = sorted(df["name"].unique())

    diet_list = ["All"] + sorted(list(df["diet"].unique()))
    selected_diet = st.sidebar.selectbox("Diet Preference:", diet_list)

    course_list = ["All"] + sorted(list(df["course"].unique()))
    selected_course = st.sidebar.selectbox("Course Preference:", course_list)

    max_price_val = int(df["price"].max()) if not df.empty else 500
    selected_price = st.sidebar.slider("Max Price (₹):", 50, max_price_val, 300)
    num_recs = st.sidebar.slider("Number of recommendations:", 5, 20, 10)

    st.sidebar.button("Logout", on_click=handle_logout_callback)

if st.session_state.logged_in and st.session_state.is_admin:
    tab_titles = ["HOME", "SYSTEM ADMIN", "RECOMMENDATION", "HISTORY", "ABOUT US"]
elif st.session_state.logged_in:
    tab_titles = ["HOME", "RECOMMENDATION", "HISTORY", "ABOUT US"]
else:
    tab_titles = ["HOME", "LOGIN/SIGN IN", "RECOMMENDATION", "HISTORY", "ABOUT US"]

if st.session_state.current_tab not in tab_titles:
    st.session_state.current_tab = "HOME"

selected_tab = st.radio(
    "Navigation",
    options=tab_titles,
    index=tab_titles.index(st.session_state.current_tab),
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
    on_change=on_tab_change,
)

if selected_tab == "HOME":
    st.subheader("FOOD MENU")

    model_live = load_data_from_file()
    df_live = model_live["df"]

    all_courses = [
        c
        for c in sorted(df_live["course"].str.title().unique())
        if str(c).strip() != ""
    ]
    max_courses = [c.title() for c in all_courses[:5]]

    if "active_course" not in st.session_state:
        st.session_state.active_course = None

    cols = st.columns(len(max_courses)) if max_courses else []
    for idx, course_title in enumerate(max_courses):
        course_raw = course_title.title()
        btn_type = (
            "primary"
            if st.session_state.active_course == course_raw
            else "secondary"
        )
        cols[idx].button(
            course_title,
            key=f"course_btn_{idx}",
            type=btn_type,
            use_container_width=True,
            on_click=set_course_callback,
            args=(course_raw,),
        )

    st.divider()
    if st.session_state.active_course is None:
        st.info("Select food type / course to view dishes.")
    else:
        selected_cat = st.session_state.active_course
        filtered_dishes = df_live[
            df_live["course"].str.title() == selected_cat
            ].reset_index(drop=True)
        total_items = len(filtered_dishes)
        st.markdown(f"**{selected_cat.title()}** : {total_items} items")

        if filtered_dishes.empty:
            st.warning("No dishes available under this category.")
        else:
            items_per_page = 10
            total_pages = math.ceil(total_items / items_per_page)
            page_key = f"dish_page_{selected_cat}"
            if page_key not in st.session_state:
                st.session_state[page_key] = 1

            if total_pages > 1:
                p_cols = st.columns([1] + [0.5] * total_pages + [2])
                for p in range(1, total_pages + 1):
                    p_btn_type = (
                        "primary"
                        if st.session_state[page_key] == p
                        else "secondary"
                    )
                    p_cols[p].button(
                        str(p),
                        key=f"page_btn_{selected_cat}_{p}",
                        type=p_btn_type,
                        on_click=set_home_page_callback,
                        args=(page_key, p),
                    )
                st.write(
                    f"Showing page **{st.session_state[page_key]}** of"
                    f" **{total_pages}**"
                )

            current_page = st.session_state.get(page_key, 1)
            start_idx = (current_page - 1) * items_per_page
            page_dishes = filtered_dishes.iloc[
                start_idx: start_idx + items_per_page
            ].reset_index(drop=True)

            top_5 = page_dishes.iloc[:5]
            if not top_5.empty:
                row1_cols = st.columns(5)
                for i, (_, row) in enumerate(top_5.iterrows()):
                    with row1_cols[i]:
                        img_to_show = (
                            row["img_url"]
                            if pd.notna(row["img_url"])
                               and str(row["img_url"]).strip()
                            else DEFAULT_IMG
                        )
                        st.image(img_to_show, use_container_width=True)
                        st.subheader(row["name"])
                        st.write(f"**Diet:** {row['diet']}")
                        st.write(f"**Price:** ₹{row['price']}")
                        st.caption(f"**Ingredients:** {row['ingredients']}")
                        st.divider()

            bottom_5 = page_dishes.iloc[5:10]
            if not bottom_5.empty:
                row2_cols = st.columns(5)
                for i, (_, row) in enumerate(bottom_5.iterrows()):
                    with row2_cols[i]:
                        img_to_show = (
                            row["img_url"]
                            if pd.notna(row["img_url"])
                               and str(row["img_url"]).strip()
                            else DEFAULT_IMG
                        )
                        st.image(img_to_show, use_container_width=True)
                        st.subheader(row["name"])
                        st.write(f"**Diet:** {row['diet']}")
                        st.write(f"**Price:** ₹{row['price']}")
                        st.caption(f"**Ingredients:** {row['ingredients']}")
                        st.divider()

elif selected_tab == "SYSTEM ADMIN":
    st.markdown("**Admin Management System**")

    admin_col1, admin_col2 = st.columns([1, 1], gap="medium")

    with admin_col1:
        admin_action = st.radio(
            "Choose Action:",
            ["ADD DISH", "UPDATE DISH", "DELETE DISH"],
            horizontal=True,
            key="admin_action_radio",
        )

        if admin_action == "ADD DISH":
            st.subheader("Add New Dish")
            with st.form("add_dish_form", clear_on_submit=True):
                new_dish_name = st.text_input("Dish Name")
                new_diet = st.selectbox(
                    "Diet",
                    ["Vegetarian", "Non-Vegetarian"],
                    placeholder="Select Diet",
                    index=None,
                )
                new_course = st.selectbox(
                    "Course",
                    [
                        "Beverage",
                        "Bread",
                        "Dessert",
                        "Main Course",
                        "Starter/Snack",
                    ],
                    placeholder="Select Course",
                    index=None,
                )
                new_price = st.number_input(
                    "Price (₹)", min_value=1, value=150, step=10
                )
                new_ingredients = st.text_area("Ingredients")
                new_img_url = st.text_input(
                    "Image URL (Optional)",
                    value=DEFAULT_IMG,
                    placeholder="Enter Image URL",
                )

                if st.form_submit_button("Add Dish"):
                    name_clean = new_dish_name.strip()
                    if not name_clean:
                        st.error("Dish name is required!")
                    elif not new_diet or not new_course:
                        st.error("Diet and Course fields are required!")
                    elif name_clean.lower() in [
                        n.lower() for n in df["name"].unique()
                    ]:
                        st.error("Dish with this name already exists!")
                    else:
                        img_to_save = (
                            new_img_url.strip()
                            if new_img_url.strip()
                            else DEFAULT_IMG
                        )
                        new_row = {
                            "name": name_clean,
                            "diet": str(new_diet).strip().title(),
                            "course": str(new_course).strip().title(),
                            "price": float(new_price),
                            "ingredients": new_ingredients.strip(),
                            "img_url": img_to_save,
                        }
                        updated_df = pd.concat(
                            [df, pd.DataFrame([new_row])], ignore_index=True
                        )
                        save_updated_df(updated_df)
                        st.success(f"'{name_clean}' added successfully!")
                        st.rerun()

        elif admin_action == "UPDATE DISH":
            st.subheader("Update Existing Dish")
            if df.empty:
                st.info("No dishes available to update.")
            else:
                dish_to_update = st.selectbox(
                    "Select Dish to Edit:",
                    options=sorted(df["name"].unique()),
                    key="select_update_dish",
                    placeholder="Dish Name",
                    index=None,
                )

                if dish_to_update:
                    selected_item = df[df["name"] == dish_to_update].iloc[0]
                    with st.form("update_dish_form"):
                        updated_name = st.text_input(
                            "Dish Name", value=str(selected_item["name"])
                        )

                        curr_diet = str(selected_item["diet"]).title()
                        diet_options = ["Vegetarian", "Non-Vegetarian"]
                        diet_idx = (
                            diet_options.index(curr_diet)
                            if curr_diet in diet_options
                            else 0
                        )

                        updated_diet = st.selectbox(
                            "Diet", diet_options, index=diet_idx
                        )
                        updated_course = st.text_input(
                            "Course", value=str(selected_item["course"]).title()
                        )
                        updated_price = st.number_input(
                            "Price (₹)",
                            min_value=1,
                            value=int(selected_item["price"]),
                        )
                        updated_ingredients = st.text_area(
                            "Ingredients",
                            value=str(selected_item["ingredients"]),
                        )

                        curr_img = (
                            str(selected_item["img_url"])
                            if pd.notna(selected_item["img_url"])
                            else ""
                        )
                        updated_img_url = st.text_input(
                            "Image URL (Optional)", value=curr_img
                        )

                        if st.form_submit_button("Save Changes"):
                            img_val = (
                                updated_img_url.strip()
                                if updated_img_url.strip()
                                else DEFAULT_IMG
                            )

                            idx = df[df["name"] == dish_to_update].index[0]
                            df.at[idx, "name"] = updated_name.strip()
                            df.at[idx, "diet"] = updated_diet.strip().title()
                            df.at[idx, "course"] = (
                                updated_course.strip().title()
                            )
                            df.at[idx, "price"] = float(updated_price)
                            df.at[idx, "ingredients"] = (
                                updated_ingredients.strip()
                            )
                            df.at[idx, "img_url"] = img_val

                            save_updated_df(df)
                            st.success(f"'{updated_name}' updated successfully!")
                            st.rerun()

        elif admin_action == "DELETE DISH":
            st.subheader("Delete Dish")
            if df.empty:
                st.info("No dishes available to delete.")
            else:
                dish_to_del = st.selectbox(
                    "Select Dish Name to Delete:",
                    options=sorted(df["name"].unique()),
                    placeholder="Dish Name",
                    index=None,
                    key="select_delete_dish",
                )

                if dish_to_del and st.button(
                        "Delete Selected Dish", type="primary"
                ):
                    updated_df = df[df["name"] != dish_to_del].reset_index(
                        drop=True
                    )
                    save_updated_df(updated_df)
                    st.success(f"'{dish_to_del}' deleted successfully!")
                    st.rerun()

    with admin_col2:
        df_latest = load_data_from_file()["df"]
        st.subheader("Current Dataset Overview")
        st.write(f"**Total Dishes:** {len(df_latest)}")
        st.dataframe(
            df_latest[["name", "diet", "course", "price"]],
            height=500,
            use_container_width=True,
        )

elif selected_tab == "LOGIN/SIGN IN":
    col1, col2 = st.columns([7, 3], border=True)
    with col1:
        st.subheader("Login Form")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            u_clean = username.strip()
            p_clean = password.strip()
            if u_clean == "" or p_clean == "":
                st.error("Enter a valid Username and Password!")
            elif u_clean == "admin" and p_clean == "admin123":
                st.session_state.logged_in = True
                st.session_state.username = "admin"
                st.session_state.full_name = "System Admin"
                st.session_state.is_admin = True
                st.session_state.current_tab = "HOME"
                st.rerun()
            else:
                user = login_user(u_clean, p_clean)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = u_clean
                    st.session_state.full_name = user[1]
                    st.session_state.is_admin = False
                    st.session_state.current_tab = "HOME"
                    st.rerun()
                else:
                    st.error(
                        "Username or Password does not exist! Please Register."
                    )
    with col2:
        st.subheader("Registration Form")
        with st.form("registration_form"):
            full_name = st.text_input("Enter Full Name")
            email = st.text_input("Email Address")
            phone = st.text_input("Phone Number")
            new_user = st.text_input("Create Username")
            new_pass = st.text_input("Create Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button("Register")

            if register_submit:
                fn_clean = full_name.strip()
                em_clean = email.strip()
                ph_clean = phone.strip()
                nu_clean = new_user.strip()
                np_clean = new_pass.strip()
                cp_clean = confirm_pass.strip()

                if (
                        not fn_clean
                        or not em_clean
                        or not ph_clean
                        or not nu_clean
                        or not np_clean
                        or not cp_clean
                ):
                    st.error("Please fill all details!")
                elif np_clean != cp_clean:
                    st.error("Passwords do not match!")
                else:
                    if register_user(
                            fn_clean, em_clean, ph_clean, nu_clean, np_clean
                    ):
                        st.session_state.logged_in = True
                        st.session_state.username = nu_clean
                        st.session_state.full_name = fn_clean
                        st.session_state.is_admin = False
                        st.session_state.current_tab = "HOME"
                        st.rerun()
                    else:
                        st.error("Username Already Exists!")

elif selected_tab == "RECOMMENDATION":
    if st.session_state.logged_in:
        selected_food = st.selectbox(
            "Choose or search a dish:",
            options=food_list,
            index=None,
            placeholder="Enter food name",
            key="rec_food_select",
        )
        if selected_food:
            selected_row = df[df["name"] == selected_food].iloc[0]
            st.subheader(f"Selected Dish: {selected_food}")
            col1, col2 = st.columns([1, 3])
            with col1:
                img_to_show = (
                    selected_row["img_url"]
                    if pd.notna(selected_row["img_url"])
                       and str(selected_row["img_url"]).strip()
                    else DEFAULT_IMG
                )
                st.image(img_to_show, use_container_width=True)
            with col2:
                st.write(f"**Diet:** {selected_row['diet']}")
                st.write(f"**Course:** {selected_row['course']}")
                st.write(f"**Price:** ₹{selected_row['price']}")
                st.write(f"**Ingredients:** {selected_row['ingredients']}")

            st.divider()
            st.subheader("Recommended For You")

            payload = {
                "dish_name": selected_food,
                "diet": selected_diet,
                "course": selected_course,
                "max_price": selected_price,
                "top_n": num_recs,
            }

            try:
                response = requests.post(FLASK_API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    results = response.json().get("recommendations", [])

                    if st.session_state.get("last_saved_dish") != selected_food:
                        save_history(
                            st.session_state.username,
                            selected_food,
                            selected_diet,
                            selected_course,
                            selected_price,
                        )
                        st.session_state.last_saved_dish = selected_food

                    if not results:
                        st.warning("No recommendations matched your filters.")
                    else:
                        cols = st.columns(5)
                        for i, item in enumerate(results):
                            with cols[i % 5]:
                                with st.container(border=True):
                                    img_rec = (
                                        item["img_url"]
                                        if item.get("img_url")
                                        else DEFAULT_IMG
                                    )
                                    st.image(img_rec, use_container_width=True)
                                    st.subheader(item["name"])
                                    st.write(f"**Diet:** {item['diet']}")
                                    st.write(f"**Price:** ₹{item['price']}")
                                    st.write(f"**Course:** {item['course']}")
                                    st.caption(
                                        f"**Ingredients:** {item['ingredients']}"
                                    )
                else:
                    st.error(f"Flask Backend API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to Flask API server. Please run 'python"
                    " api.py' on port 5000!"
                )
    else:
        st.warning("Please LOGIN to access recommendations.")

elif selected_tab == "HISTORY":
    if not st.session_state.logged_in:
        st.warning("Please LOGIN to view your recommendation history.")
    else:
        history_df = get_history(st.session_state.username)

        h_col1, h_col2 = st.columns([8, 2])
        with h_col1:
            st.subheader("History of Viewed Dishes")
        with h_col2:
            if not history_df.empty:
                st.button(
                    "Clear All History",
                    type="secondary",
                    on_click=handle_clear_all_callback,
                    args=(st.session_state.username,),
                )

        if history_df.empty:
            st.info("No recommendation history recorded yet!")
        else:
            history_dish_names = history_df["selected_food"].tolist()

            matched_dishes = df[df["name"].isin(history_dish_names)].copy()
            matched_dishes["name"] = pd.Categorical(
                matched_dishes["name"],
                categories=history_dish_names,
                ordered=True,
            )
            matched_dishes = matched_dishes.sort_values("name").reset_index(
                drop=True
            )

            dishes_list = matched_dishes.to_dict("records")
            total_history_items = len(dishes_list)

            items_per_page = 5
            total_history_pages = math.ceil(
                total_history_items / items_per_page
            )

            if "history_page" not in st.session_state:
                st.session_state.history_page = 1

            if st.session_state.history_page > total_history_pages:
                st.session_state.history_page = max(1, total_history_pages)

            if total_history_pages > 1:
                p_cols = st.columns([1] + [0.5] * total_history_pages + [2])
                for p in range(1, total_history_pages + 1):
                    p_btn_type = (
                        "primary"
                        if st.session_state.history_page == p
                        else "secondary"
                    )
                    p_cols[p].button(
                        str(p),
                        key=f"hist_page_btn_{p}",
                        type=p_btn_type,
                        on_click=set_history_page_callback,
                        args=(p,),
                    )
                st.write(
                    f"Showing page **{st.session_state.history_page}** of"
                    f" **{total_history_pages}**"
                )

            start_idx = (st.session_state.history_page - 1) * items_per_page
            current_page_dishes = dishes_list[
                start_idx: start_idx + items_per_page
            ]

            cols = st.columns(5)
            for idx, d in enumerate(current_page_dishes):
                with cols[idx]:
                    with st.container(border=True):
                        img_url = (
                            d.get("img_url") if d.get("img_url") else DEFAULT_IMG
                        )
                        st.image(img_url, use_container_width=True)
                        st.markdown(f"**{d.get('name', 'Dish')}**")
                        st.write(f"**Price:** ₹{d.get('price', 0)}")
                        st.caption(f"**Course:** {d.get('course', '')}")

                        st.button(
                            "Remove",
                            key=f"rem_hist_{start_idx + idx}_{d.get('name')}",
                            type="secondary",
                            use_container_width=True,
                            on_click=handle_remove_single_callback,
                            args=(st.session_state.username, d.get("name")),
                        )

            st.divider()
            st.subheader("Recommended Based on Your History")

            FLASK_HISTORY_API_URL = "http://127.0.0.1:5000/recommend_history"
            payload = {"history_dishes": history_dish_names, "top_n": 20}
            try:
                response = requests.post(
                    FLASK_HISTORY_API_URL, json=payload, timeout=5
                )
                if response.status_code == 200:
                    history_recs = response.json().get("recommendations", [])
                    if not history_recs:
                        st.info("No personalized history recommendations found.")
                    else:
                        render_side_slider(history_recs)
                else:
                    st.error(f"Flask Backend API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to Flask API server. Please run 'python"
                    " api.py' on port 5000!"
                )

elif selected_tab == "ABOUT US":
    st.markdown("""
    ### Food Recommendation System

    This project uses:
    - Python programming language
    - Backend: Flask REST API (api.py)
    - Frontend: Streamlit (app.py)
    - Train Model: (train.py - model.pkl)
        - TF-IDF Vectorization
        - Cosine Similarity
        - Content-Based Filtering

    Developed to recommend dishes based on similar ingredients and food characteristics.

    The application uses a CSV file named: **food_dataset.csv**

    Total Number of items: **214+**

    The Dataset is Referred from Kaggle and Further Preprocessed using Gemini, some Images Added Manually and 
    Extended by System Admin.

    Dataset link - https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101 
    """)