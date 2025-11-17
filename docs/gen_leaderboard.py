"""Generate combined leaderboard page for BEHAVIOR-1K challenge."""

import json
from pathlib import Path
import mkdocs_gen_files

def load_submissions(track_dir):
    """Load all submissions from a track directory."""
    submissions = []
    track_path = Path("docs/challenge_submissions") / track_dir
    
    if not track_path.exists():
        return []
    
    for json_file in track_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            assert "overall_scores" in data, "Missing overall_scores in submission JSON"
            
            submission = {
                "team": data.get("team", "Unknown"),
                "affiliation": data.get("affiliation", ""),
                "date": data.get("date", ""),
                "overall_q_score": data["overall_scores"].get("q_score", 0),
                "overall_time_score": data["overall_scores"].get("time_score", 0),
                "overall_base_distance_score": data["overall_scores"].get("base_distance_score", 0),
                "overall_left_distance_score": data["overall_scores"].get("left_distance_score", 0),
                "overall_right_distance_score": data["overall_scores"].get("right_distance_score", 0),
            }
            submissions.append(submission)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    # Sort by average success rate (descending)
    submissions.sort(key=lambda x: x['overall_q_score'], reverse=True)
    return submissions

def generate_combined_leaderboard():
    """Generate a single leaderboard page with both tracks."""
    
    tracks = {
        "Standard Track": "standard",
        "Privileged Track": "privileged"
    }
    
    with mkdocs_gen_files.open("challenge/leaderboard.md", "w") as fd:
        
        fd.write("# Challenge Leaderboards\n\n")
        
        for track_name, track_dir in tracks.items():
            submissions = load_submissions(track_dir)
            
            fd.write(f"## {track_name}\n\n")
            
            if not submissions:
                fd.write("No submissions yet. Be the first to submit!\n\n")
            else:
                # Leaderboard table
                fd.write("| Rank | Team | Affiliation | Date | Q Score (🢁) <br/> (self-reported) | Time Score (🢁) <br/> (self-reported) | Distance Score (🢁) <br/> (self-reported) |\n")
                fd.write("|------|------|-------------|------|-----------------------------|---------------------------------|------------------------------------|\n")

                for i, sub in enumerate(submissions, 1):
                    q_score = f"{sub['overall_q_score']:.4f}"
                    time_score = f"{sub['overall_time_score']:.4f}"
                    distance_score = f"{((sub['overall_base_distance_score'] + sub['overall_left_distance_score'] + sub['overall_right_distance_score']) / 3):.4f}"
                    fd.write(f"| {i} | {sub['team']} | {sub['affiliation']} | {sub['date']} | {q_score} | {time_score} | {distance_score} |\n")

                fd.write("\n")


# Generate the leaderboard when this module is imported during mkdocs build
generate_combined_leaderboard()