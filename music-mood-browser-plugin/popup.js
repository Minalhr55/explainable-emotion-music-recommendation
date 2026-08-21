// First, map 6 mood categories to coordinates
const MOOD_MAP = {
  excited_elated:  { label: 'Excited / Elated',  valence: 0.75,  arousal: 0.75  },
  angry_stressed:  { label: 'Angry / Stressed',  valence: -0.75, arousal: 0.75  },
  sad_depressed:   { label: 'Sad / Depressed',   valence: -0.75, arousal: -0.75 },
  bored_tired:     { label: 'Bored / Tired',     valence: -0.50, arousal: -0.50 },
  calm_relaxed:    { label: 'Calm / Relaxed',    valence: 0.75,  arousal: -0.75 },
  content_serene:  { label: 'Content / Serene',  valence: 0.85,  arousal: -0.65 },
};

// Song logging function
async function logSong(quadrantKey) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab) {
        document.getElementById('status').innerText = "No active tab found!";
        return;
    }

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: getTrackInfo
    },
    (results) => {
        if (!results || !results[0] || !results[0].result) {
            document.getElementById('status').innerText = "Open Spotify/YouTube!";
            return;
        }

        const trackData = results[0].result;
        const spec = MOOD_MAP[quadrantKey];

        if (!spec) {
            console.error(`Invalid quadrant key: ${quadrantKey}`);
            return;
        }

        const record = {
            timestamp: new Date().toISOString(),
            platform: tab.url && tab.url.includes("spotify") ? "Spotify" : "YouTube",
            track_title: trackData.title,
            artist: trackData.artist,
            quadrant: quadrantKey, // Fix: Stores 'excited_elated', etc. matching process_plugin_data()
            target_valence: spec.valence,
            target_arousal: spec.arousal
        };

        // Locally store track record
        chrome.storage.local.get({ logged_tracks: [] }, (data) => {
            const updated = data.logged_tracks;
            updated.push(record);
            chrome.storage.local.set({ logged_tracks: updated }, () => {
                document.getElementById('status').innerText = `Logged: ${trackData.title}`;
            });
        });
    });
}

function getTrackInfo() {
    let title = "", artist = "";
    if (window.location.hostname.includes("spotify.com")) {
        title = document.querySelector('[data-testid="context-item-link"]')?.innerText || document.title;
        artist = document.querySelector('[data-testid="context-item-info-artist"]')?.innerText || "Unknown artist";
    } else if (window.location.hostname.includes("youtube.com")) {
        title = document.querySelector('h1.ytd-watch-metadata')?.innerText || document.title;
        artist = document.querySelector('#owner #channel-name')?.innerText || "YouTube Video";
    }
    return { title: title.trim(), artist: artist.trim() };
}

// Event listeners for all 6 mood buttons
// Ensure button IDs in popup.html match these exact 6 keys:
Object.keys(MOOD_MAP).forEach(key => {
    const btn = document.getElementById(key);
    if (btn) {
        btn.addEventListener('click', () => logSong(key));
    }
});

// Exporting to CSV
document.getElementById('export')?.addEventListener('click', () => {
    chrome.storage.local.get({ logged_tracks: [] }, (data) => {
        if (!data.logged_tracks || data.logged_tracks.length === 0) {
            alert("No tracks logged yet");
            return;
        }
        let CSV = "Timestamp,Platform,Track,Artist,Quadrant,Target_Valence,Target_Arousal\n";
        data.logged_tracks.forEach(r => {
            CSV += `"${r.timestamp}","${r.platform}","${r.track_title.replace(/"/g, '""')}","${r.artist.replace(/"/g, '""')}","${r.quadrant}",${r.target_valence},${r.target_arousal}\n`;
        });
        const blob = new Blob([CSV], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = "Plugin_Music_Mood_Data.csv";
        a.click();
    });
});