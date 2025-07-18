import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm

# TODO: incorporate mq_ef and skill type annotation once format is finalized
# TODO: add auto-QA to check if the annotation make sense

def create_annotated_video(video_path: str, json_path: str, output_path: str, panel_width: int = 400):
    """Create annotated video with side panel."""
    
    # Load annotations
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    skills = annotations.get('skill_annotation', [])
    primitives = annotations.get('primitive_annotation', [])
    
    # Setup video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width + panel_width, orig_height))
    
    frame_num = 0
    
    # Progress bar with tqdm
    with tqdm(total=total_frames, desc="Processing frames", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find current annotations
            current_skill = next((s for s in skills if s['frame_duration'][0] <= frame_num <= s['frame_duration'][1]), None)
            current_primitive = next((p for p in primitives if p['frame_duration'][0] <= frame_num <= p['frame_duration'][1]), None)
            
            # Create info panel
            panel = np.ones((orig_height, panel_width, 3), dtype=np.uint8) * 255
            cv2.rectangle(panel, (0, 0), (panel_width-1, orig_height-1), (200, 200, 200), 2)
            
            # Draw text helper
            y = 30
            margin = 10
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            def add_text(text: str, color=(80, 80, 80), font_scale=0.5, bold=False):
                nonlocal y
                thickness = 2 if bold else 1
                # Simple word wrap
                words = text.split(' ')
                line = ""
                for word in words:
                    test_line = f"{line} {word}".strip()
                    if cv2.getTextSize(test_line, font, font_scale, thickness)[0][0] < panel_width - 20:
                        line = test_line
                    else:
                        if line:
                            cv2.putText(panel, line, (margin, y), font, font_scale, color, thickness, cv2.LINE_AA)
                            y += 20
                        line = word
                if line:
                    cv2.putText(panel, line, (margin, y), font, font_scale, color, thickness, cv2.LINE_AA)
                    y += 25
            
            def format_objects(obj_list):
                if not obj_list:
                    return "None"
                flat = []
                for obj in obj_list:
                    if isinstance(obj, list):
                        flat.extend(obj)
                    else:
                        flat.append(obj)
                return ", ".join(list(dict.fromkeys(flat)))  # Keep full names, dedupe
            
            # Panel content
            add_text("ANNOTATION", (30, 30, 30), 0.6, True)
            y += 10
            
            if current_primitive:
                add_text("PRIMITIVE:", (20, 100, 20), 0.55, True)
                desc = ", ".join(current_primitive.get('primitive_description', ['Composite Action']))
                add_text(f"Action: {desc}")
                add_text(f"Objects: {format_objects(current_primitive.get('object_id', []))}")
                start, end = current_primitive['frame_duration']
                add_text(f"Duration: {start}-{end} ({end-start}f)")
                y += 10
            
            if current_skill:
                add_text("SKILL:", (100, 20, 20), 0.55, True)
                desc = ", ".join(current_skill.get('skill_description', ['Unknown']))
                add_text(f"Action: {desc}")
                add_text(f"Objects: {format_objects(current_skill.get('object_id', []))}")
                add_text(f"Manipulating: {format_objects(current_skill.get('manipulating_object_id', []))}")
                
                spatial = current_skill.get('spatial_prefix', [])
                spatial_text = ", ".join(str(sp) for sp in spatial) if spatial else "None"
                add_text(f"Spatial: {spatial_text}")
                
                skill_idx = current_skill.get('skill_idx', 'N/A')
                start, end = current_skill['frame_duration']
                add_text(f"ID: #{skill_idx} ({start}-{end})")
            elif current_primitive:
                add_text("SKILL: Transition")
            else:
                add_text("STATUS: No annotation")
            
            # Frame number at bottom
            cv2.putText(panel, f"Frame: {frame_num}", (margin, orig_height - 20), 
                       font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
            
            # Combine panel and video
            combined = np.zeros((orig_height, orig_width + panel_width, 3), dtype=np.uint8)
            combined[:, :panel_width] = panel
            combined[:, panel_width:] = frame
            
            out.write(combined)
            frame_num += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Add annotations to MP4 video')
    parser.add_argument('video_path', help='Input video file path')
    parser.add_argument('json_path', help='Annotation JSON file path')  
    parser.add_argument('output_path', help='Output video file path')
    parser.add_argument('--panel-width', type=int, default=400, help='Width of annotation panel (default: 400)')
    
    args = parser.parse_args()
    
    try:
        create_annotated_video(args.video_path, args.json_path, args.output_path, args.panel_width)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()