import streamlit as st
import joblib
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from spotify_client import SpotifyMoodClient

st.set_page_config(page_title="MOODIFY", page_icon="🎵")

st.markdown("""
<style>
    .stButton button {
        background-color: #1DB954;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "mood_model.joblib"
    data = joblib.load(model_path)
    return data["vectorizer"], data["model"]

@st.cache_resource
def load_spotify():
    return SpotifyMoodClient()

def count_sentences(text: str) -> int:
    import re
    matches = re.findall(r"[.!?]+(?=\s|$)", text.strip())
    return len(matches) if matches else 1

def main():
    st.title("🎵MOODIFY")

    try:
        vectorizer, model = load_model()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    try:
        spotify = load_spotify()
    except Exception as e:
        st.error(f"Spotify init failed: {e}")
        st.stop()

    user_input = st.text_input(
        "How are you feeling?",
        placeholder="e.g., I'm feeling great today!",
        help="Preferably one or two sentence for best accuracy",
        max_chars=150,
    )

    if st.button("Get Songs"):
        if not user_input.strip():
            st.warning("Please write something!")
            st.stop()

        sentence_count = count_sentences(user_input)
        if sentence_count > 2:
            st.warning(
                f"⚠️ You wrote {sentence_count} sentences. Accuracy may be lower."
            )

        # Predict mood
        text_vec = vectorizer.transform([user_input])
        mood = model.predict(text_vec)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(text_vec)[0]
            confidence = float(probas.max())

        # Store in session
        st.session_state.user_input = user_input
        st.session_state.mood = mood
        st.session_state.confidence = confidence
        st.session_state.final_mood = mood
        st.session_state.show = True
        st.session_state.manual_override = False  # Reset override flag

    if st.session_state.get("show"):
        final_mood = st.session_state.get("final_mood", st.session_state.mood)
        conf = st.session_state.get("confidence")

        # Show mood prediction
        if conf is not None and conf < 0.39:
            # Low confidence: show both metrics + auto dropdown
            col1, col2 = st.columns(2)
            col1.metric("Mood", str(st.session_state.mood).upper())
            col2.metric("Confidence", f"{conf:.0%}")
            
            st.warning("Not sure. Pick your mood:")
            final_mood = st.selectbox(
                "Select mood:",
                list(model.classes_),
                index=list(model.classes_).index(final_mood),
                key="final_mood_select",
            )
            st.session_state.final_mood = final_mood
            st.session_state.manual_override = True
        else:
            # High confidence: show only mood + wrong button
            if not st.session_state.get("manual_override"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success(f"**Detected Mood:** {str(final_mood).upper()}")
                with col2:
                    if st.button("❌ Wrong?"):
                        st.session_state.manual_override = True
                        st.rerun()
            
            # If user clicked "Wrong?", show dropdown
            if st.session_state.get("manual_override"):
                st.info("Pick the correct mood:")
                final_mood = st.selectbox(
                    "Your actual mood:",
                    list(model.classes_),
                    index=list(model.classes_).index(st.session_state.mood),
                    key="manual_mood_select",
                )
                st.session_state.final_mood = final_mood

        st.markdown("---")

        try:
            songs = spotify.recommend_songs(final_mood, seed_text=st.session_state.user_input, n=5)
        except Exception as e:
            st.error(f"Spotify error: {e}")
            st.stop()

        st.subheader(f"Songs for {str(final_mood).upper()} mood:")

        for i, song in enumerate(songs, 1):
            cols = st.columns([1, 4])

            img = song.get("image")
            if img:
                cols[0].image(img, use_container_width=True)
            else:
                cols[0].write("")

            cols[1].markdown(f"**{i}. {song.get('name','')}**  \n{song.get('artist','')}")
            if song.get("url"):
                cols[1].markdown(f"[Listen on Spotify]({song['url']})")

if __name__ == "__main__":
    main()