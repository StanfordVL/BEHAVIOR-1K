"""Generate combined leaderboard page for BEHAVIOR-1K challenge."""

import json
from pathlib import Path
import mkdocs_gen_files

def load_submissions():
    """Load all submissions from a track directory."""
    submissions = []
    track_path = Path("docs/challenge_submissions")
    
    if not track_path.exists():
        return []
    
    for json_file in track_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            assert "overall_scores" in data, "Missing overall_scores in submission JSON"
            
            submission = {
                "team": data.get("team", "Unknown"),
                "affiliation": data.get("affiliation", "Unknown"),
                "date": data.get("date", ""),
                "track": data.get("track", ""),
                "overall_q_score": data["overall_scores"].get("q_score", 0),
                "overall_task_sr": data["overall_scores"].get("task_sr", 0),
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
    """Generate a single leaderboard page."""
    
    with mkdocs_gen_files.open("challenge/leaderboard.md", "w") as fd:
        
        fd.write("# Challenge Leaderboards\n\n")
        fd.write(
            '!!! warning "Provisional 2025 Challenge Leaderboard"\n'
            "    The entries below are submissions to the 2025 BEHAVIOR challenge. "
            "All results shown are self-reported on the public validation set and have "
            "not yet been verified on the hidden test set. While we work through "
            "evaluating submissions on the hidden test set, we are surfacing the "
            "self-reported scores to provide a rough sense of current performance. "
            "We will update the leaderboard with scores from the hidden test instances "
            "as evaluations complete.\n\n"
        )
        fd.write(
            '!!! info "About Q-score"\n'
            "    We rank policies by Q-score. Q-score measures how much of a task's goal "
            "condition a policy satisfies by computing the fraction of completed "
            "sub-goals and choosing the best-matched goal clause. It awards partial "
            "credit, so policies that make meaningful progress score higher even without "
            "full completion. This makes Q-score a smoother, more reliable way to "
            "compare policies across BEHAVIOR tasks than a binary success rate.\n\n"
        )
        fd.write(
            '!!! banner "Submission Tracks"\n'
            "    All standard track submissions for the 2025 BEHAVIOR challenge "
            "will automatically be considered for prizes in the privileged track as well.\n\n"
        )
        
        submissions = load_submissions()
        
        # fd.write(f"## {track_name}\n\n")
        
        if not submissions:
            fd.write("No submissions yet. Be the first to submit!\n\n")
        else:
            # Leaderboard table
            fd.write("| Rank | Team | Affiliation | Date | Track | Full Task Success Rate <br/> (self-reported) | :material-star: **Q Score** <br/> (self-reported)\n")
            fd.write("|------|------|-------------|------|-------|----------------------------------------------|---------------------------------------------|\n")

            for i, sub in enumerate(submissions, 1):
                q_score = f"{sub['overall_q_score']:.4f}"
                task_sr = f"{sub['overall_task_sr']:.4f}"
                fd.write(
                    f"| — | {sub['team']} | {sub['affiliation']} | {sub['date']} | "
                    f"{sub['track'].capitalize()} | {task_sr} | **{q_score}** |\n"
                )
            
            fd.write("\n")


# Generate the leaderboard when this module is imported during mkdocs build
generate_combined_leaderboard()