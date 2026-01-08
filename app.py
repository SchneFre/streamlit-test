import streamlit as st
from backend import create_lineplot, create_barplot, create_dataframe, MovieRecommender

def main():
    st.title('Saklia Dashboard')

    # Sidebar navigation
    st.sidebar.header('Navigation')
    page = st.sidebar.radio(
        "Go to",
        ["Home", "EDA", "Prediction"]
    )


    # ------------------ Pages ------------------

    if page == "Home":
        st.subheader("Welcome to the Dashboard")
        # Add image from URL
        st.image(
            "https://camo.githubusercontent.com/89b13def8ccb5217f1ee985f419d201cbe29f50518d54124ff8e425937351f74/68747470733a2f2f656475636174696f6e2d7465616d2d323032302e73332d65752d776573742d312e616d617a6f6e6177732e636f6d2f646174612d616e616c79746963732f64617461626173652d73616b696c612d736368656d612e706e67",
            caption="Sakila Database",
            use_container_width=True
        )

    elif page == "EDA":
        st.subheader("Daily rentals by store")
        plt = create_lineplot()
        st.pyplot(plt)
        st.subheader("Total benefit by each store")
        plt = create_barplot()
        st.pyplot(plt)

        st.subheader("top five most rented movies by each store in 2005")
        df = create_dataframe()
        st.dataframe(df)

    elif page == "Prediction":
        # Initialize recommender
        recommender = MovieRecommender(csv_path="movies.csv")

        st.title("Movie Description Similarity Search")

        # User text input
        user_input = st.text_area(
            "Enter a movie description to find similar films:",
            height=150,
            placeholder="Type a description here..."
        )

        # Button
        if st.button("Get Your Prediction"):
            if user_input.strip() == "":
                st.warning("Please enter a description.")
            else:
                # Get top 3 similar movies from backend
                results = recommender.get_similar_movies(user_input, top_n=3)

                st.subheader("Top 3 Most Similar Movies")
                for r in results:
                    st.write(f"**{r['title']}** (Rating: {r['rating']}) — Similarity: {r['similarity']:.4f}")
if __name__ == '__main__':
    main()
