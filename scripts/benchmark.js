const canvasDict = {
	'baseline_total_canvas': 		["Total frame time", 		["Empty scene, flatcache on", "Rs_int, flatcache on", "Rs_int, with 1 Fetch robot, flatcache on", "Rs_int, with 3 Fetch robot, flatcache on"]],
	'baseline_physics_canvas': 		["Physics step time", 		["Empty scene, flatcache on", "Rs_int, flatcache on", "Rs_int, with 1 Fetch robot, flatcache on", "Rs_int, with 3 Fetch robot, flatcache on"]],
	'baseline_rendering_canvas': 	["Render step time", 		["Empty scene, flatcache on", "Rs_int, flatcache on", "Rs_int, with 1 Fetch robot, flatcache on", "Rs_int, with 3 Fetch robot, flatcache on"]],
	'baseline_non_physics_canvas': 	["Non-physics step time",	["Empty scene, flatcache on", "Rs_int, flatcache on", "Rs_int, with 1 Fetch robot, flatcache on", "Rs_int, with 3 Fetch robot, flatcache on"]],
	'np_total_canvas': 				["Total frame time", 		["Rs_int, with 1 Fetch robot, fluids", "Rs_int, with 1 Fetch robot, cloth", "Rs_int, with 1 Fetch robot, macro particles", "Rs_int, with 1 Fetch robot, cloth, fluids, macro particles"]],
	'np_physics_canvas': 			["Physics step time",   	["Rs_int, with 1 Fetch robot, fluids", "Rs_int, with 1 Fetch robot, cloth", "Rs_int, with 1 Fetch robot, macro particles", "Rs_int, with 1 Fetch robot, cloth, fluids, macro particles"]],
	'np_rendering_canvas': 			["Render step time", 		["Rs_int, with 1 Fetch robot, fluids", "Rs_int, with 1 Fetch robot, cloth", "Rs_int, with 1 Fetch robot, macro particles", "Rs_int, with 1 Fetch robot, cloth, fluids, macro particles"]],
	'np_non_physics_canvas': 		["Non-physics step time", 	["Rs_int, with 1 Fetch robot, fluids", "Rs_int, with 1 Fetch robot, cloth", "Rs_int, with 1 Fetch robot, macro particles", "Rs_int, with 1 Fetch robot, cloth, fluids, macro particles"]],
	'scene_total_canvas': 			["Total frame time", 		["Rs_int, with 1 Fetch robot, flatcache on", "Rs_int, with 1 Fetch robot, flatcache on", "Rs_int, with 1 Fetch robot, flatcache on", "Rs_int, with 1 Fetch robot, flatcache on"]],
	'scene_physics_canvas': 		["Physics step time", 		["Rs_int, with 1 Fetch robot, fluids", "Rs_int, with 1 Fetch robot, cloth", "Rs_int, with 1 Fetch robot, macro particles", "Rs_int, with 1 Fetch robot, cloth, fluids, macro particles"]],
	'scene_rendering_canvas': 		["Render step time", 		["Rs_int, with 1 Fetch robot, fluids", "Rs_int, with 1 Fetch robot, cloth", "Rs_int, with 1 Fetch robot, macro particles", "Rs_int, with 1 Fetch robot, cloth, fluids, macro particles"]],
	'scene_non_physics_canvas': 	["Non-physics step time", 	["Rs_int, with 1 Fetch robot, fluids", "Rs_int, with 1 Fetch robot, cloth", "Rs_int, with 1 Fetch robot, macro particles", "Rs_int, with 1 Fetch robot, cloth, fluids, macro particles"]],
}



$('#baseline_tab a').on('click', function (e) {
    e.preventDefault()
    $(this).tab('show')
})

$('#np_tab a').on('click', function (e) {
    e.preventDefault()
    $(this).tab('show')
})

$('#scene_tab a').on('click', function (e) {
    e.preventDefault()
    $(this).tab('show')
})

function init() {
	function collectBenchesPerTestCase(entries) {
		const map = new Map();
		for (const entry of entries) {
			const {commit, date, tool, benches} = entry;
			for (const bench of benches) {
				const result = { commit, date, tool, bench };
				const title_map = map.get(bench.extra);
				if (title_map === undefined) {
					const temp_map = new Map();
					temp_map.set(bench.name, [result]);
					map.set(bench.extra, temp_map);
				} else {
					const name_map = title_map.get(bench.name);
					if (name_map === undefined) {
						title_map.set(bench.name, [result]);
					} else {
						name_map.push(result);
					}
				}
			}
		}
		return map;
	}

	const data = window.BENCHMARK_DATA;

	// Render header
	document.getElementById('last-update').textContent = new Date(data.lastUpdate).toString();
	const repoLink = document.getElementById('repository-link');
	repoLink.href = data.repoUrl;
	repoLink.textContent = data.repoUrl;

	// Render footer
	document.getElementById('dl-button').onclick = () => {
		const dataUrl = 'data:,' + JSON.stringify(data, null, 2);
		const a = document.createElement('a');
		a.href = dataUrl;
		a.download = 'benchmark_data.json';
		a.click();
	};

	// Prepare data points for charts
	return collectBenchesPerTestCase(data.entries['Benchmark']);
}


function renderGraph(canvasName, fieldName, runNames) {
	// get filtered data
	let filteredData = new Map(Array.from(allData.get(fieldName)).filter(([key, _value]) => {
		return runNames.includes(key);
	}));
	const canvas = document.getElementById(canvasName);
	const color = '#38ff38';
	const data = {
		labels: Array.from(filteredData).map(([_name, value]) => (value[0].commit.id.slice(0, 7))),
		datasets: Array.from(filteredData).map(([name, value]) => ({
			label: name,
			data: value.map(d => d.bench.value),
			borderColor: color,
			backgroundColor: 'rgba(0, 0, 0, 0.01)'
		}))
	};
	const options = {
		tooltips: {
			callbacks: {
				afterTitle: items => {
					const {index} = items[0];
					const data = filteredData.values().next().value[index];
					return '\n' + data.commit.message + '\n\n' + data.commit.timestamp + ' committed by @' + data.commit.committer.username + '\n';
				},
				label: item => {
					let label = item.value;
					const { range, unit } = filteredData.values().next().value[item.index].bench;
					label += ' ' + unit;
					if (range) {
						label += ' (' + range + ')';
					}
					return label;
				},
				afterLabel: item => {
					const { extra } = filteredData.values().next().value[item.index].bench;
					return extra ? '\n' + extra : '';
				}
			}
		},
		onClick: (_mouseEvent, activeElems) => {
			if (activeElems.length === 0) {
				return;
			}
			// XXX: Undocumented. How can we know the index?
			const index = activeElems[0]._index;
			const url = filteredData.values().next().value[index].commit.url;
			window.open(url, '_blank');
		},
		title: {
			display: true,
			text: fieldName,
		},
		layout: {
		  	padding: 0
		},
		responsive: true,
		maintainAspectRatio: true
	};

	new Chart(canvas, {
		type: 'line',
		data,
		options,
	});
}



const allData = init()
for (const [canvasName, [fieldName, runNames]] of Object.entries(canvasDict)) {
	renderGraph(canvasName, fieldName, runNames);
}

