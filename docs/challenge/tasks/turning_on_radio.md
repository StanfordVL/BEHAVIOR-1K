---
icon: material/video-outline
---

# Task 0: Turning On Radio

**Rooms:** Living Room  
**Duration:** 45s avg  
**Task Definition:** [View on BEHAVIOR Knowledge Base](https://behavior.stanford.edu/knowledgebase/tasks/turning_on_radio-0.html)

=== "RGB"
    
    <div class="video-container">
      <iframe src="https://player.vimeo.com/video/1109198872?title=0&byline=0&portrait=0&badge=0&autopause=0&player_id=0&app_id=58479" frameborder="0" allowfullscreen></iframe>
    </div>

=== "Depth"

    <div class="video-container">
      <iframe src="https://player.vimeo.com/video/1109198677?title=0&byline=0&portrait=0&badge=0&autopause=0&player_id=0&app_id=58479" frameborder="0" allowfullscreen></iframe>
    </div>

=== "Segmentation"

    <div class="video-container">
      <iframe src="https://player.vimeo.com/video/1109198928?title=0&byline=0&portrait=0&badge=0&autopause=0&player_id=0&app_id=58479" frameborder="0" allowfullscreen></iframe>
    </div>

=== "Bounding Box"

    <div class="video-container">
      <iframe src="https://player.vimeo.com/video/1109198811?title=0&byline=0&portrait=0&badge=0&autopause=0&player_id=0&app_id=58479" frameborder="0" allowfullscreen></iframe>
    </div>

<div class="annotation-sidebar">
  <div class="annotation-header">Current Step</div>
  <div class="annotation-content" id="annotation-display">
    <div class="annotation-description">—</div>
    <div class="annotation-objects">
      <div class="objects-label">Objects:</div>
      <ul class="objects-list"></ul>
    </div>
    <div class="annotation-time">—</div>
  </div>
</div>

<script src="https://player.vimeo.com/api/player.js"></script>
<script>
// Annotation data embedded directly
const annotationData = {
    "skill_annotation": [
        {
            "skill_description": ["move to"],
            "object_id": [["radio_89"]],
            "memory_prefix": [],
            "frame_duration": [55, 300]
        },
        {
            "skill_description": ["pick up from"],
            "object_id": [["radio_89", "coffee_table_koagbh_0"]],
            "memory_prefix": [],
            "frame_duration": [301, 641]
        },
        {
            "skill_description": ["press"],
            "object_id": [["radio_89"]],
            "memory_prefix": [],
            "frame_duration": [642, 900]
        },
        {
            "skill_description": ["place on"],
            "object_id": [["radio_89", "coffee_table_koagbh_0"]],
            "memory_prefix": ["back"],
            "frame_duration": [900, 1245]
        }
    ]
};

// Process annotations for easier lookup
const FPS = 30;
const annotations = annotationData.skill_annotation.map(skill => ({
    start: skill.frame_duration[0] / FPS,
    end: skill.frame_duration[1] / FPS,
    description: skill.memory_prefix.length > 0 
        ? `${skill.skill_description[0]} ${skill.memory_prefix.join(' ')}`
        : skill.skill_description[0],
    objects: skill.object_id[0]
}));

// Simplify object IDs for display
function simplifyObjectId(id) {
    // Remove numbers and underscores, make readable
    return id.replace(/_/g, ' ')
             .replace(/\b[0-9]+\b/g, '')
             .replace(/\s+/g, ' ')
             .trim()
             .replace(/\b\w/g, l => l.toUpperCase());
}

// Format time display
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Update annotation display
function updateAnnotation(currentTime) {
    const current = annotations.find(a => currentTime >= a.start && currentTime < a.end);
    
    const descEl = document.querySelector('.annotation-description');
    const listEl = document.querySelector('.objects-list');
    const timeEl = document.querySelector('.annotation-time');
    
    if (current) {
        descEl.textContent = current.description;
        
        listEl.innerHTML = '';
        current.objects.forEach(obj => {
            const li = document.createElement('li');
            li.textContent = simplifyObjectId(obj);
            listEl.appendChild(li);
        });
        
        timeEl.textContent = `${formatTime(current.start)} - ${formatTime(current.end)}`;
    } else {
        descEl.textContent = '—';
        listEl.innerHTML = '';
        timeEl.textContent = '—';
    }
}

// Initialize Vimeo players when page loads
document.addEventListener('DOMContentLoaded', function() {
    const iframes = document.querySelectorAll('.video-container iframe');
    const players = [];
    
    iframes.forEach(iframe => {
        const player = new Vimeo.Player(iframe);
        players.push(player);
        
        // Listen to time updates from any player
        player.on('timeupdate', function(data) {
            updateAnnotation(data.seconds);
        });
    });
});
</script>

<style>
/* Position sidebar absolutely within parent container */
.md-content__inner {
  position: relative;
}

/* Video container adjustments - square aspect ratio */
.tabbed-set {
  max-width: 500px; /* Limit video width for square format */
}

.video-container {
  position: relative;
  padding-bottom: 100%; /* 1:1 square aspect ratio */
  height: 0;
  background: #000;
}

.video-container iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

/* Annotation sidebar - positioned absolutely */
.annotation-sidebar {
  position: absolute;
  top: 280px; /* Aligned with video tabs */
  left: 540px; /* Position to the right of the video */
  width: 280px;
  background: var(--md-code-bg-color);
  border-radius: 8px;
  padding: 1.5rem;
  height: fit-content;
}

.annotation-header {
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--md-default-fg-color--light);
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
}

.annotation-content {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.annotation-description {
  font-size: 1.25rem;
  font-weight: 500;
  color: var(--md-default-fg-color); /* Changed from primary color to default for better contrast */
  line-height: 1.4;
}

.annotation-objects {
  margin-top: 0.5rem;
}

.objects-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--md-default-fg-color--light);
  margin-bottom: 0.5rem;
}

.objects-list {
  margin: 0;
  padding-left: 1.25rem;
  list-style: none;
}

.objects-list li {
  position: relative;
  color: var(--md-default-fg-color);
  font-size: 0.95rem;
  line-height: 1.6;
  padding-left: 0.5rem;
}

.objects-list li::before {
  content: "•";
  position: absolute;
  left: -0.5rem;
  color: var(--md-accent-fg-color);
}

.annotation-time {
  font-size: 0.875rem;
  color: var(--md-default-fg-color--light);
  padding-top: 0.75rem;
  margin-top: 0.5rem;
  border-top: 1px solid var(--md-default-fg-color--lightest);
}

/* Responsive design */
@media (max-width: 900px) {
  .tabbed-set {
    max-width: 100%;
  }
  
  .annotation-sidebar {
    position: static;
    left: auto;
    top: auto;
    width: 100%;
    margin-top: 2rem;
  }
}

/* Hide the sidebar on very small screens */
@media (max-width: 600px) {
  .annotation-sidebar {
    display: none;
  }
}
</style>