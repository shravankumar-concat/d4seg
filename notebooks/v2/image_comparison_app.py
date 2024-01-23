import streamlit as st
from PIL import Image
import os
import json
from datetime import datetime

def load_image(directory, filename, target_size=(300, 300)):
    image_file_name, ext = os.path.splitext(filename)
    image_path = os.path.join(directory, filename)

    try:
        image_path = os.path.join(directory, f"{image_file_name}.png")
        image = Image.open(image_path)
    except Exception as e:
        image_path = os.path.join(directory, f"{image_file_name}.jpg")
        image = Image.open(image_path)

    # Resize the image to the target size
    image = image.resize(target_size)
    return image

def get_session_id():
    try:
        session_id = st.session_state.report_id
    except:
        session_id = id(st)
        st.session_state.report_id = session_id
    return session_id

def main():
    st.title("Segmentation Model Evaluation")
    
    # Get user's name
    user_name = st.text_input("Enter your name:")

    base_dir = "/home/shravan/documents/deeplearning/github/alpha_matte_segmentation/notebooks/v2/Review_Dir"
    list_dirs = os.listdir(base_dir)
    selected_dir = st.selectbox("Select Image Directory:", list_dirs)

    st.subheader(f"Directory: {selected_dir}")

    dir1, dir2, dir3, dir4 = [
        f"{base_dir}/{selected_dir}/Original",
        f"{base_dir}/{selected_dir}/InHouse",
        f"{base_dir}/{selected_dir}/rembg",
        f"{base_dir}/{selected_dir}/model_out"
    ]

    filenames = [f for f in os.listdir(dir1) if not f.startswith('.')]

    # Initialize variables to store ratings
    results_dict = {}
    
    # Get the session ID
    session_id = get_session_id()

    # User details and timestamp
    user_details = {
        "user_id": session_id,
        "user_name": user_name,
        "timestamp": str(datetime.now()),
    }    

    # Check if results.json already exists
    json_filename = f"{user_name}_results.json"
    if os.path.isfile(json_filename):
        with open(json_filename, 'r') as json_file:
            results_dict = json.load(json_file)


    
    for selected_filename in filenames:
        st.subheader(f"Image: {selected_filename}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image(load_image(dir1, selected_filename), caption="Directory 1", use_column_width=True)
            rank_radio1 = st.radio("Rank this image", ["Not Ranked", "Rank 1"], key=f"radio1_{selected_filename}")

        with col2:
            st.image(load_image(dir2, selected_filename), caption="Directory 2", use_column_width=True)
            rank_radio2 = st.radio("Rank this image", ["Not Ranked", "Rank 2"], key=f"radio2_{selected_filename}")

        with col3:
            st.image(load_image(dir3, selected_filename), caption="Directory 3", use_column_width=True)
            rank_radio3 = st.radio("Rank this image", ["Not Ranked", "Rank 3"], key=f"radio3_{selected_filename}")

        with col4:
            st.image(load_image(dir4, selected_filename), caption="Directory 4", use_column_width=True)
            rank_radio4 = st.radio("Rank this image", ["Not Ranked", "Rank 4"], key=f"radio4_{selected_filename}")

        # Collect user ranking
        radio_values = [rank_radio1, rank_radio2, rank_radio3, rank_radio4]
        selected_rank = [i + 1 for i, value in enumerate(radio_values) if value.startswith("Rank")]

        if selected_rank:
            selected_model = selected_rank[0]
            ranked_models = ['Original', 'InHouse', 'Rembg', 'D4Seg']
            ranked_model = ranked_models[selected_model - 1]
        else:
            selected_model = None
            ranked_model = None

        # Save the results to the dictionary
        if selected_dir not in results_dict:
            results_dict[selected_dir] = {"user_details": user_details, "data": {}}
        results_dict[selected_dir]["data"][selected_filename] = {"Model": ranked_model, "Ranking": selected_model}

    # Save the updated results dictionary to a single JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(results_dict, json_file)        

    # Print the dictionary containing evaluation results
    st.write(f"Evaluation Results Dictionary for {selected_dir}:", results_dict)
    st.write(f"Results saved to {json_filename}")

if __name__ == "__main__":
    main()
