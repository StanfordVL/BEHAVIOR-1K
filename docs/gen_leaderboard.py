"""Generate combined leaderboard page for BEHAVIOR-1K challenge."""

import yaml
from pathlib import Path
import mkdocs_gen_files

def load_submissions(track_dir):
    """Load all submissions from a track directory."""
    submissions = []
    track_path = Path("docs/challenge_submissions") / track_dir
    
    if not track_path.exists():
        return []
    
    for yaml_file in track_path.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            
            # Calculate overall success rate
            if 'results' in data:
                total_rate = sum(task['success_rate'] for task in data['results'])
                avg_rate = total_rate / len(data['results']) if data['results'] else 0
            else:
                avg_rate = 0
                
            submission = {
                'team': data.get('team', 'Unknown'),
                'affiliation': data.get('affiliation', ''),
                'date': data.get('date', ''),
                'avg_success_rate': avg_rate,
                'results': data.get('results', [])
            }
            submissions.append(submission)
            
        except Exception as e:
            print(f"Error loading {yaml_file}: {e}")
    
    # Sort by average success rate (descending)
    submissions.sort(key=lambda x: x['avg_success_rate'], reverse=True)
    return submissions

def generate_combined_leaderboard():
    """Generate a single leaderboard page with both tracks."""
    
    tracks = {
        "Standard Track": "standard_track",
        "Privileged Information Track": "privileged_track"
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
                fd.write("| Rank | Team | Affiliation | Success Rate | Date |\n")
                fd.write("|------|------|-------------|--------------|------|\n")
                
                for i, sub in enumerate(submissions, 1):
                    rate_percent = f"{sub['avg_success_rate']:.1%}"
                    fd.write(f"| {i} | {sub['team']} | {sub['affiliation']} | {rate_percent} | {sub['date']} |\n")
                
                fd.write("\n")
        
        # Submission instructions
        fd.write("## How to Submit\n\n")
        fd.write("To submit your results to the leaderboard:\n\n")
        fd.write("1. **Submit self-reported scores** through this [google form](#)\n")
        fd.write("2. **Wait for review** - once approved, your results will appear on the leaderboard!\n\n")

# Generate the leaderboard when this module is imported during mkdocs build
generate_combined_leaderboard()