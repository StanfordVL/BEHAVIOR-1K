---
icon: material/grid
---

# 2025 BEHAVIOR Challenge Demo Gallery

Browse through all 50 household tasks in our 2025 challenge. Click on any task to view RGB, depth, segmentation, bounding box, and language annotation visualizations.

<div class="controls">
  <div class="filter-control">
    <label for="room-filter">Filter by room:</label>
    <select id="room-filter">
      <option value="all">All Rooms</option>
      <option value="kitchen">Kitchen</option>
      <option value="living-room">Living Room</option>
      <option value="bedroom">Bedroom</option>
      <option value="bathroom">Bathroom</option>
      <option value="dining-room">Dining Room</option>
      <option value="garage">Garage</option>
    </select>
  </div>
  
  <div class="sort-control">
    <label for="sort-select">Sort by:</label>
    <select id="sort-select">
      <option value="name">Task Name</option>
      <option value="duration-asc">Duration (Short → Long)</option>
      <option value="duration-desc">Duration (Long → Short)</option>
    </select>
  </div>
</div>

<div class="grid cards compact" id="task-grid"></div>

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
  background: var(--md-default-bg-color);
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 8px;
  padding: 1rem;
  transition: transform 0.2s, box-shadow 0.2s;
}

.task-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.task-card.hidden {
  display: none;
}

.task-thumbnail {
  width: 100%;
  border-radius: 4px;
  margin-bottom: 0.75rem;
}

.task-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: var(--md-default-fg-color);
}

.task-meta {
  color: var(--md-default-fg-color--light);
  font-size: 0.9rem;
  margin: 0.25rem 0;
}

.task-link {
  display: inline-block;
  margin-top: 0.75rem;
  color: var(--md-primary-fg-color);
  text-decoration: none;
  font-weight: 500;
}

.task-link:hover {
  text-decoration: underline;
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

<script>
(function() {
  // Centralized task data
  const tasks = [
    {
      id: 'turning_on_radio',
      name: 'Turning On Radio',
      rooms: ['living-room'],
      duration: 45,
      thumbnail: 'https://i.vimeocdn.com/video/2046655249-86934e83f1a8908d7650d7d7794ce5b7a6303dabbb8182e286623b17614c13bf-d_295x166',
      path: './tasks/turning_on_radio.md'
    },
    {
      id: 'picking_up_trash',
      name: 'Picking Up Trash',
      rooms: ['kitchen', 'living-room'],
      duration: 90,
      thumbnail: 'https://i.vimeocdn.com/video/2046660576-1b8dc9c40d1e4a390fd7ddc851da0c62895371e0d27a52c4133e9d8391ce28ca-d_295x166',
      path: './tasks/picking_up_trash.md'
    },
    {
      id: 'putting_away_halloween_decorations',
      name: 'Putting Away Halloween Decorations',
      rooms: ['living-room'],
      duration: 120,
      thumbnail: 'https://i.vimeocdn.com/video/2046668931-9db3428391e4025c251343a4a026a6ead0afb888965b6a3987fdecd6785ece76-d_295x166',
      path: './tasks/putting_away_Halloween_decorations.md'
    }
    // TODO: Add remaining 47 tasks here
  ];

  // Room display names
  const roomNames = {
    'kitchen': 'Kitchen',
    'living-room': 'Living Room',
    'bedroom': 'Bedroom',
    'bathroom': 'Bathroom',
    'dining-room': 'Dining Room',
    'garage': 'Garage'
  };

  // Initialize gallery - runs immediately
  function initGallery() {
    const taskGrid = document.getElementById('task-grid');
    const roomFilter = document.getElementById('room-filter');
    const sortSelect = document.getElementById('sort-select');
    
    if (!taskGrid || !roomFilter || !sortSelect) {
      // Elements not ready, try again
      setTimeout(initGallery, 10);
      return;
    }
    
    let currentTasks = [...tasks];
    
    // Render tasks
    function renderTasks(taskList) {
      taskGrid.innerHTML = '';
      
      taskList.forEach(task => {
        const card = document.createElement('div');
        card.className = 'task-card';
        card.dataset.id = task.id;
        
        const roomsDisplay = task.rooms.map(r => roomNames[r]).join(', ');
        
        card.innerHTML = `
          <img src="${task.thumbnail}" alt="${task.name}" class="task-thumbnail">
          <div class="task-title">${task.name}</div>
          <div class="task-meta">📍 ${roomsDisplay}</div>
          <div class="task-meta">⏱️ ${task.duration}s avg</div>
          <a href="${task.path}" class="task-link">View Task →</a>
        `;
        
        taskGrid.appendChild(card);
      });
    }
    
    // Filter tasks
    function filterTasks() {
      const selectedRoom = roomFilter.value;
      
      if (selectedRoom === 'all') {
        currentTasks = [...tasks];
      } else {
        currentTasks = tasks.filter(task => task.rooms.includes(selectedRoom));
      }
      
      sortTasks();
    }
    
    // Sort tasks
    function sortTasks() {
      const sortBy = sortSelect.value;
      
      currentTasks.sort((a, b) => {
        switch(sortBy) {
          case 'name':
            return a.name.localeCompare(b.name);
          case 'duration-asc':
            return a.duration - b.duration;
          case 'duration-desc':
            return b.duration - a.duration;
          default:
            return 0;
        }
      });
      
      renderTasks(currentTasks);
    }
    
    // Event listeners
    roomFilter.addEventListener('change', filterTasks);
    sortSelect.addEventListener('change', sortTasks);
    
    // Initial render
    renderTasks(currentTasks);
  }
  
  // Start initialization immediately
  initGallery();
})();
</script>