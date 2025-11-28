"""Generate combined leaderboard page for BEHAVIOR-1K challenge."""

import json
from collections import OrderedDict
from pathlib import Path
import mkdocs_gen_files

def load_submissions():
    """Load all submissions from a track directory."""
    submissions = dict()
    track_path = Path("docs/challenge_submissions")
    
    if not track_path.exists():
        return []
    
    for json_file in track_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            assert "overall_scores" in data, "Missing overall_scores in submission JSON"
            team = data.get("team")
            testset = data.get("testset")
            submission = {
                "affiliation": data.get("affiliation", "Unknown"),
                "date": data.get("date", ""),
                "track": data.get("track", ""),
                f"{testset}_q_score": data["overall_scores"].get("q_score", 0),
                f"{testset}_task_sr": data["overall_scores"].get("task_sr", 0),
                f"{testset}_time_score": data["overall_scores"].get("time_score", 0),
                f"{testset}_base_distance_score": data["overall_scores"].get("base_distance_score", 0),
                f"{testset}_left_distance_score": data["overall_scores"].get("left_distance_score", 0),
                f"{testset}_right_distance_score": data["overall_scores"].get("right_distance_score", 0),
            }
            if team not in submissions:
                submissions[team] = submission
            else:
                submissions[team].update(submission)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    # Now, sort submissions by hidden_q_score descending, if no hidden q_score, use public_q_score
    # Also, those with a hidden score should rank higher than those without
    submissions = OrderedDict(
        sorted(
            submissions.items(),
            key=lambda item: (item[1].get("hidden_q_score", -1), item[1].get("public_q_score", -1)),
            reverse=True,
        )
    )
    return submissions

def generate_combined_leaderboard():
    """Generate a single leaderboard page."""
    
    with mkdocs_gen_files.open("challenge/leaderboard.md", "w") as fd:
        
        fd.write("# Challenge Leaderboards\n\n")
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
            '!!! banner "Submission tracks & test sets"\n'
            "    This page lists submissions to the 2025 BEHAVIOR challenge across all tracks and test sets.\n"
            "    Standard-track submissions remain eligible for privileged-track consideration where applicable.\n\n"
            "    'Validation (Public)' entries are self-reported on the public validation set.\n"
            "    'Held-out Test (Hidden)' entries are from the hidden test set and are verified by the BEHAVIOR team.\n\n"
        )
        
        submissions = load_submissions()
        if not submissions:
            fd.write("No submissions yet. Be the first to submit!\n\n")
        else:
            # HTML Leaderboard table with hidden/public subcolumns
            fd.write(
                "<table>\n"
                "  <thead>\n"
                "    <tr>\n"
                "      <th rowspan=\"2\">Rank</th>\n"
                "      <th rowspan=\"2\">Team</th>\n"
                "      <th rowspan=\"2\">Affiliation</th>\n"
                "      <th rowspan=\"2\">Date</th>\n"
                "      <th rowspan=\"2\">Track</th>\n"
                "      <th colspan=\"2\">Full Task Success Rate</th>\n"
                "      <th colspan=\"2\">★ <strong>Q Score</strong></th>\n"
                "    </tr>\n"
                "    <tr>\n"
                "      <th>Validation (Public)</th>\n"
                "      <th>Held-out Test (Hidden)</th>\n"
                "      <th>Validation (Public)</th>\n"
                "      <th>Held-out Test (Hidden)</th>\n"
                "    </tr>\n"
                "  </thead>\n"
                "  <tbody>\n"
            )

            for i, (team, sub) in enumerate(submissions.items(), 1):
                public_q_score = f"{sub.get('public_q_score', 0):.4f}"
                hidden_q_score = f"{sub['hidden_q_score']:.4f}" if "hidden_q_score" in sub else ""
                public_task_sr = f"{sub.get('public_task_sr', 0):.4f}"
                hidden_task_sr = f"{sub['hidden_task_sr']:.4f}" if "hidden_task_sr" in sub else ""
                fd.write(
                    "    <tr>"
                    f"<td>{i}</td>"
                    f"<td>{team}</td>"
                    f"<td>{sub.get('affiliation', '')}</td>"
                    f"<td>{sub.get('date', '')}</td>"
                    f"<td>{sub.get('track', '').capitalize()}</td>"
                    f"<td>{public_task_sr}</td>"
                    f"<td>{hidden_task_sr}</td>"
                    f"<td>{public_q_score}</td>"
                    f"<td><strong>{hidden_q_score}</strong></td>"
                    "</tr>\n"
                )

            fd.write("  </tbody>\n</table>\n\n")


# Generate the leaderboard when this module is imported during mkdocs build
generate_combined_leaderboard()