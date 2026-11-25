import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
import hashlib
import os
from dotenv import load_dotenv

class SpotifyMoodClient:
    MOOD_PLAYLISTS = {
        'happy': '0JbMs91ZaHcTLnLulwfgsS',
        'sad': '3ri4Utp9Ncuj8AzpsBFPhM',
        'angry': '4b7HTwBChNEzOuqJOe1bVa',
        'love': '0f6FlaKNCJSOZakeZFgkAa',
        'neutral': '6UGNH1FTGhMCxWeGQ1g3fI'
    }

    def __init__(self):
        load_dotenv()

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.getenv('SPOTIFY_CLIENT_ID'),
            client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
            redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI'),
            scope="playlist-read-private playlist-read-collaborative"
        ))

    def get_playlist_tracks(self, playlist_id):
        tracks = []

        results = self.sp.playlist_tracks(playlist_id)
        while results:
            for item in results.get('items', []):
                track = item.get('track')
                if not track:
                    continue

                images = (track.get('album') or {}).get('images') or []
                image_url = images[0]['url'] if len(images) > 0 else None

                artists = track.get("artists", []) or []
                artist_names = [a.get("name") for a in artists if a and a.get("name")]
                artist_str = ", ".join(artist_names) if artist_names else "Unknown artist"

                tracks.append({
                    "name": track.get("name") or "Unknown track",
                    "artist": artist_str,
                    "url": (track.get("external_urls") or {}).get("spotify"),
                    "image": image_url,
})

            results = self.sp.next(results) if results.get('next') else None

        return tracks

    def recommend_songs(self, mood, seed_text=None, n=5):
        playlist_id = self.MOOD_PLAYLISTS.get(mood)
        if not playlist_id:
            raise ValueError(f"Unknown mood '{mood}'. Available moods: {list(self.MOOD_PLAYLISTS.keys())}")

        all_tracks = self.get_playlist_tracks(playlist_id)

        if seed_text:
            seed = int(hashlib.md5(seed_text.encode()).hexdigest(), 16) % (2**32)
            random.seed(seed)

        return random.sample(all_tracks, min(n, len(all_tracks)))

if __name__ == "__main__":
    client = SpotifyMoodClient()
    songs = client.recommend_songs('happy', "test", n=5)

    for song in songs:
        print(f"{song['name']} - {song['artist']} | image: {song['image']}")
