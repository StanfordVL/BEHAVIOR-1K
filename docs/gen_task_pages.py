"""Generate task pages for the challenge demo galleries.

Generates the demo gallery (index + one page per task) for each challenge year from its
committed ``task_data.json``. The 2025 challenge has 50 tasks; the 2026 challenge has 100
(the 50 from 2025 plus 50 new ones). Tasks without a ``video``/``duration``/``thumbnail``
field render with "coming soon" placeholders.
"""

import json
from pathlib import Path

import mkdocs_gen_files

# Room display names (shared across years).
ROOM_NAMES = {
    "kitchen": "Kitchen",
    "living_room": "Living Room",
    "bedroom": "Bedroom",
    "bathroom": "Bathroom",
    "garage": "Garage",
    "garden": "Garden",
    "childs_room": "Child's Room",
    "corridor": "Corridor",
    "utility_room": "Utility Room",
    "dining_room": "Dining Room",
    "entryway": "Entryway",
    "private_office": "Private Office",
    "shared_office": "Shared Office",
    "copy_room": "Copy Room",
    "bar": "Bar",
}

# Galleries to generate: (output prefix under the site, source task_data.json, intro blurb).
GALLERIES = [
    (
        "challenge",
        Path("docs/challenge/task_data.json"),
        "Browse through all 100 household tasks in our 2026 challenge (the 50 tasks from 2025 "
        "plus 50 new ones). Click on any task to view an example of RGB video demonstration "
        "where available.",
    ),
    (
        "challenge/archive/2025",
        Path("docs/challenge/archive/2025/task_data.json"),
        "Browse through all 50 household tasks in our 2025 challenge. Click on any task to "
        "view an example of RGB video demonstration.",
    ),
]

# Filter dropdown <option> list, built from ROOM_NAMES.
ROOM_FILTER_OPTIONS = "\n".join(
    f'      <option value="{key}">{label}</option>' for key, label in ROOM_NAMES.items()
)

# JS room-name map mirroring ROOM_NAMES.
ROOM_NAMES_JS = ",\n".join(f"    {json.dumps(key)}: {json.dumps(label)}" for key, label in ROOM_NAMES.items())

GALLERY_CONTROLS = f"""<div class="controls">
  <div class="filter-control">
    <label for="room-filter">Filter by room:</label>
    <select id="room-filter">
      <option value="all">All Rooms</option>
{ROOM_FILTER_OPTIONS}
    </select>
  </div>

  <div class="sort-control">
    <label for="sort-select">Sort by:</label>
    <select id="sort-select">
      <option value="index" selected>Task Number</option>
      <option value="name">Task Name</option>
      <option value="duration-asc">Duration (Short → Long)</option>
      <option value="duration-desc">Duration (Long → Short)</option>
    </select>
  </div>
</div>

<div class="grid cards compact" id="task-grid">
  <div class="loading">Loading tasks...</div>
</div>
"""

GALLERY_SCRIPT_SUFFIX = f"""

  // Room display names
  const roomNames = {{
{ROOM_NAMES_JS}
  }};

  let currentTasks = [...tasks];

  // Initialize gallery
  function initGallery() {{
    const taskGrid = document.getElementById('task-grid');
    const roomFilter = document.getElementById('room-filter');
    const sortSelect = document.getElementById('sort-select');

    if (!taskGrid || !roomFilter || !sortSelect) {{
      setTimeout(initGallery, 10);
      return;
    }}

    // Render tasks
    function renderTasks(taskList) {{
      taskGrid.innerHTML = '';

      taskList.forEach((task) => {{
        const card = document.createElement('a');
        card.className = 'task-card';
        // Get original task index from the full tasks array
        const taskIndex = tasks.indexOf(task);
        const taskIndexPadded = String(taskIndex).padStart(2, '0');
        card.href = `./${{taskIndexPadded}}_${{task.id}}.html`;
        card.dataset.id = task.id;

        const roomsDisplay = task.rooms.map(r => roomNames[r] || r).join(', ');

        // Create thumbnail element
        let thumbnailHtml;
        if (task.thumbnail) {{
          thumbnailHtml = `<img src="${{task.thumbnail}}" alt="${{task.name}}" class="task-thumbnail">`;
        }} else {{
          thumbnailHtml = `<div class="task-thumbnail placeholder">📹</div>`;
        }}

        // Format duration
        let durationDisplay = '';
        if (task.duration) {{
          const minutes = Math.floor(task.duration / 60);
          const seconds = task.duration % 60;
          if (minutes === 0) {{
            durationDisplay = `${{seconds}}s`;
          }} else if (seconds === 0) {{
            durationDisplay = `${{minutes}}m`;
          }} else {{
            durationDisplay = `${{minutes}}m ${{seconds}}s`;
          }}
        }}

        card.innerHTML = `
          ${{thumbnailHtml}}
          <div class="task-number">Task ${{taskIndex}}</div>
          <div class="task-title">${{task.name}}</div>
          <div class="task-metadata">
            <span class="task-room">${{roomsDisplay}}</span>
            <span class="task-duration">${{durationDisplay}}</span>
          </div>
        `;

        taskGrid.appendChild(card);
      }});
    }}

    // Filter tasks
    function filterTasks() {{
      const selectedRoom = roomFilter.value;

      if (selectedRoom === 'all') {{
        currentTasks = [...tasks];
      }} else {{
        currentTasks = tasks.filter(task => task.rooms.includes(selectedRoom));
      }}

      sortTasks();
    }}

    // Sort tasks
    function sortTasks() {{
      const sortBy = sortSelect.value;

      currentTasks.sort((a, b) => {{
        switch(sortBy) {{
          case 'index':
            // Sort by original task index
            return tasks.indexOf(a) - tasks.indexOf(b);
          case 'name':
            return a.name.localeCompare(b.name);
          case 'duration-asc':
            return (a.duration || 0) - (b.duration || 0);
          case 'duration-desc':
            return (b.duration || 0) - (a.duration || 0);
          default:
            return 0;
        }}
      }});

      renderTasks(currentTasks);
    }}

    // Event listeners
    roomFilter.addEventListener('change', filterTasks);
    sortSelect.addEventListener('change', sortTasks);

    // Initial render
    renderTasks(currentTasks);
  }}

  // Start initialization
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', initGallery);
  }} else {{
    initGallery();
  }}
}})();
</script>
"""

GALLERY_STYLE = """
<style>
.controls {
  display: flex;
  gap: 2rem;
  margin: 1.5rem 0;
  align-items: center;
  padding: 1rem;
  background: var(--md-code-bg-color);
  border-radius: 8px;
}

.filter-control, .sort-control {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.controls label {
  font-weight: 500;
  color: var(--md-default-fg-color);
  white-space: nowrap;
}

#room-filter, #sort-select {
  padding: 0.5rem;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 4px;
  background: var(--md-default-bg-color);
  color: var(--md-default-fg-color);
  cursor: pointer;
}

.grid.cards.compact {
  display: grid !important;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)) !important;
  gap: 1rem;
  margin-top: 1.5rem;
}

.task-card {
  display: flex;
  flex-direction: column;
  background: var(--md-default-bg-color);
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 8px;
  padding: 1rem;
  transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
  position: relative;
  text-decoration: none;
  color: inherit;
  height: 100%;
}

.task-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  border-color: var(--md-primary-fg-color);
  text-decoration: none;
}

.task-card:hover .task-metadata {
  opacity: 1;
}

.task-number {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  backdrop-filter: blur(4px);
}

.task-thumbnail {
  width: 100%;
  border-radius: 4px;
  margin-bottom: 0.75rem;
  aspect-ratio: 16/9;
  object-fit: cover;
  background: var(--md-code-bg-color);
}

.task-thumbnail.placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--md-default-fg-color--light);
  font-size: 48px;
}

.task-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--md-default-fg-color);
  line-height: 1.3;
  transition: color 0.2s;
}

.task-card:hover .task-title {
  text-decoration: underline;
}

.task-metadata {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;
  padding-top: 0.5rem;
  border-top: 1px solid var(--md-default-fg-color--lightest);
  font-size: 0.7rem;
  color: var(--md-default-fg-color--light);
  opacity: 0.65;
  transition: opacity 0.2s;
}

.task-room {
  display: flex;
  align-items: center;
  gap: 0.2rem;
}

.task-room::before {
  content: "📍";
  font-size: 0.7rem;
  opacity: 0.7;
}

.task-duration {
  display: flex;
  align-items: center;
  gap: 0.2rem;
  font-weight: 500;
}

.task-duration::before {
  content: "⏱";
  font-size: 0.7rem;
  opacity: 0.7;
}

.loading {
  text-align: center;
  padding: 2rem;
  color: var(--md-default-fg-color--light);
}

@media (max-width: 768px) {
  .controls {
    flex-direction: column;
    align-items: stretch;
    gap: 1rem;
  }

  .grid.cards.compact {
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)) !important;
  }
}
</style>
"""

TASK_VIDEO_STYLE = """<style>
/* Video wrapper for proper sizing */
.video-wrapper {
  max-width: 720px;
  margin: 2rem 0;
}

.video-wrapper iframe {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 1/1; /* Square video */
}

/* Video placeholder */
.video-placeholder {
  width: 720px;
  max-width: 100%;
  aspect-ratio: 1/1;
  background: var(--md-code-bg-color);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.placeholder-content {
  text-align: center;
  color: var(--md-default-fg-color--light);
}

.placeholder-content i {
  font-size: 64px;
  margin-bottom: 1rem;
}

/* Responsive design */
@media (max-width: 768px) {
  .video-wrapper,
  .video-placeholder {
    max-width: 100%;
  }
}
</style>
"""


def generate_gallery(prefix, task_data_file, blurb):
    """Generate the demo gallery index + per-task pages for one challenge year."""
    with open(task_data_file) as f:
        data = json.load(f)

    # Also copy the task_data.json file to the output.
    with mkdocs_gen_files.open(f"{prefix}/task_data.json", "w") as fd:
        json.dump(data, fd, indent=2)

    # Create the demo gallery as index page with all tasks embedded.
    with mkdocs_gen_files.open(f"{prefix}/tasks/index.md", "w") as fd:
        fd.write("# Demo Gallery\n\n")
        fd.write(blurb + "\n\n")

        # Controls + grid.
        fd.write(GALLERY_CONTROLS)

        # JavaScript with embedded data.
        fd.write("\n<script>\n")
        fd.write("(function() {\n")
        fd.write("  // Embedded task data\n")
        fd.write("  const tasks = ")
        fd.write(json.dumps(data["tasks"], indent=2))
        fd.write(";\n")
        fd.write(GALLERY_SCRIPT_SUFFIX)

        # Styles.
        fd.write(GALLERY_STYLE)

    # Generate individual task pages (without annotations, with proper video sizing).
    for task in data["tasks"]:
        task_id = task["id"]
        task_name = task["name"]
        task_index = data["tasks"].index(task)

        # File path with zero-padded task number prefix for proper sorting.
        doc_path = Path(prefix, "tasks", f"{task_index:02d}_{task_id}.md")

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            # Page header.
            fd.write("---\n")
            fd.write("icon: material/video-outline\n")
            fd.write("---\n\n")

            # Title.
            fd.write(f"# Task {task_index}: {task_name}\n\n")

            # Metadata.
            rooms_display = ", ".join([ROOM_NAMES.get(r, r.title()) for r in task.get("rooms", [])])
            duration = task.get("duration", "N/A")

            # Format duration as "x minutes y seconds".
            if isinstance(duration, int):
                minutes = duration // 60
                seconds = duration % 60
                if minutes == 0:
                    duration_display = f"{seconds} seconds"
                elif seconds == 0:
                    duration_display = f"{minutes} minutes"
                else:
                    duration_display = f"{minutes} minutes {seconds} seconds"
            else:
                duration_display = str(duration)

            fd.write(f"**Rooms:** {rooms_display}  \n")
            fd.write(f"**Duration:** {duration_display} avg  \n")

            # Add task instruction if available.
            if task.get("instruction"):
                fd.write(f"**Language Instruction:** {task['instruction']}  \n")

            # Link to BEHAVIOR knowledge base (if available).
            kb_url = f"https://behavior.stanford.edu/knowledgebase/tasks/{task_id}-0.html"
            fd.write(f"**Full Task Definition:** [View on BEHAVIOR Knowledge Base]({kb_url})\n\n")

            # Video section - only RGB with proper sizing and minimal controls.
            if task.get("video"):
                video_url = task["video"]
                if "vimeo.com" in video_url:
                    # Minimal Vimeo params: controls on; hide title/byline/portrait/sidedock/logo; dnt.
                    fd.write('<div class="video-wrapper">\n')
                    fd.write(
                        f'  <iframe src="{video_url}?controls=1&title=0&byline=0&portrait=0&dnt=1&transparent=0&sidedock=0&logo=0" '
                    )
                    fd.write(
                        'width="720" height="720" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>\n'
                    )
                    fd.write("</div>\n\n")
            else:
                # Placeholder when video is not available.
                fd.write('<div class="video-placeholder">\n')
                fd.write('  <div class="placeholder-content">\n')
                fd.write('    <i class="material-icons">videocam_off</i>\n')
                fd.write("    <p>Video demonstration coming soon</p>\n")
                fd.write("  </div>\n")
                fd.write("</div>\n\n")

            # Styles for video.
            fd.write(TASK_VIDEO_STYLE)

        # Set edit path for the generated file.
        mkdocs_gen_files.set_edit_path(doc_path, Path(f"../../docs/{prefix}/task_data.json"))


for _prefix, _task_data_file, _blurb in GALLERIES:
    generate_gallery(_prefix, _task_data_file, _blurb)
