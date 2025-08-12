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

<div class="grid cards compact" id="task-grid" markdown>

- ![Turning On Radio](https://i.vimeocdn.com/video/2046655249-86934e83f1a8908d7650d7d7794ce5b7a6303dabbb8182e286623b17614c13bf-d_295x166){ .task-thumbnail }

    **Turning On Radio**
    {: .task-title data-rooms="living-room" data-duration="45" }

    ---

    :material-home: Living Room  
    :material-clock-outline: 45s avg

    [:octicons-arrow-right-24: View Task](./tasks/turning_on_radio.md)

- ![Picking Up Trash](https://i.vimeocdn.com/video/2046660576-1b8dc9c40d1e4a390fd7ddc851da0c62895371e0d27a52c4133e9d8391ce28ca-d_295x166){ .task-thumbnail }

    **Picking Up Trash**
    {: .task-title data-rooms="kitchen living-room" data-duration="90" }

    ---

    :material-home: Kitchen, Living Room  
    :material-clock-outline: 90s avg

    [:octicons-arrow-right-24: View Task](./tasks/picking_up_trash.md)

- ![Putting Away Halloween Decorations](https://i.vimeocdn.com/video/2046668931-9db3428391e4025c251343a4a026a6ead0afb888965b6a3987fdecd6785ece76-d_295x166){ .task-thumbnail }

    **Putting Away Halloween Decorations**
    {: .task-title data-rooms="living-room" data-duration="120" }

    ---

    :material-home: Living Room
    :material-clock-outline: 120s avg

    [:octicons-arrow-right-24: View Task](./tasks/putting_away_Halloween_decorations.md)

<!-- TODO: Add remaining 47 task cards in the same format -->

</div>

<style>
/* Controls styling */
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

/* Make grid cards more compact to fit more per row */
.grid.cards.compact {
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)) !important;
}

.grid.cards.compact > li {
  margin: 0.5rem !important;
  list-style: none !important;  /* Remove the dots */
}

.grid.cards.compact > li::before {
  display: none !important;  /* Remove any pseudo-element markers */
}

.grid.cards.compact .task-thumbnail {
  width: 100%;
  border-radius: 4px;
  margin-bottom: 0.5rem;
}

/* Reduce padding in cards for compact view */
.grid.cards.compact > li > div {
  padding: 1rem !important;
}

.grid.cards.compact > li > div > p {
  margin: 0.5rem 0 !important;
  font-size: 0.9rem;
}

/* Ensure grid stays consistent when sorting */
.grid.cards {
  display: grid !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  #search-input {
    max-width: 100%;
  }
  
  .grid.cards.compact {
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)) !important;
  }
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const taskGrid = document.getElementById('task-grid');
  const roomFilter = document.getElementById('room-filter');
  const sortSelect = document.getElementById('sort-select');

  function initializeCards() {
    const taskCards = Array.from(taskGrid.querySelectorAll('li'));
    
    // Extract data from each card
    taskCards.forEach(card => {
      // Get task name from the bold text
      const strongElement = card.querySelector('strong');
      if (strongElement) {
        card.dataset.name = strongElement.textContent.trim().toLowerCase();
      }
      
      // Extract duration from text content
      const durationMatch = card.textContent.match(/(\d+)s avg/);
      if (durationMatch) {
        card.dataset.duration = durationMatch[1];
      }
      
      // Extract rooms from the content - look for the room names in the text
      const textContent = card.textContent.toLowerCase();
      const rooms = [];
      
      if (textContent.includes('kitchen')) rooms.push('kitchen');
      if (textContent.includes('living room')) rooms.push('living-room');
      if (textContent.includes('bedroom')) rooms.push('bedroom');
      if (textContent.includes('bathroom')) rooms.push('bathroom');
      if (textContent.includes('dining room')) rooms.push('dining-room');
      if (textContent.includes('garage')) rooms.push('garage');
      
      card.dataset.rooms = rooms.join(' ');
    });

    return taskCards;
  }

  // Initialize cards
  const taskCards = initializeCards();

  // Filter functionality
  function applyFilter() {
    const selectedRoom = roomFilter.value;
    
    taskCards.forEach(card => {
      if (selectedRoom === 'all') {
        card.style.display = '';
      } else {
        const cardRooms = card.dataset.rooms || '';
        const hasRoom = cardRooms.includes(selectedRoom);
        card.style.display = hasRoom ? '' : 'none';
      }
    });
  }

  // Sort functionality  
  function applySort() {
    const sortBy = sortSelect.value;
    
    const sortedCards = [...taskCards].sort((a, b) => {
      switch(sortBy) {
        case 'name':
          return (a.dataset.name || '').localeCompare(b.dataset.name || '');
        case 'duration-asc':
          return parseInt(a.dataset.duration || '0') - parseInt(b.dataset.duration || '0');
        case 'duration-desc':
          return parseInt(b.dataset.duration || '0') - parseInt(a.dataset.duration || '0');
        default:
          return 0;
      }
    });

    // Clear and re-add sorted cards
    taskCards.forEach(card => card.remove());
    sortedCards.forEach(card => taskGrid.appendChild(card));
  }

  // Event listeners
  roomFilter.addEventListener('change', applyFilter);
  sortSelect.addEventListener('change', applySort);
});
</script>