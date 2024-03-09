const canvasDict = {
	'baseline_total_canvas': 		["FPS",				 		["Empty scene", "Rs_int", "Rs_int, with 1 Fetch", "Rs_int, with 3 Fetch"]],
	'baseline_loading_canvas': 		["Loading time", 			["Empty scene", "Rs_int", "Rs_int, with 1 Fetch", "Rs_int, with 3 Fetch"]],
	'baseline_omni_canvas': 		["Omni step time", 			["Empty scene", "Rs_int", "Rs_int, with 1 Fetch", "Rs_int, with 3 Fetch"]],
	'baseline_non_omni_canvas':		["Non-omni step time",		["Empty scene", "Rs_int", "Rs_int, with 1 Fetch", "Rs_int, with 3 Fetch"]],
	'baseline_mem_canvas': 			["Memory usage",			["Empty scene", "Rs_int", "Rs_int, with 1 Fetch", "Rs_int, with 3 Fetch"]],
	'baseline_vram_canvas': 		["Vram usage",				["Empty scene", "Rs_int", "Rs_int, with 1 Fetch", "Rs_int, with 3 Fetch"]],
	'np_total_canvas': 				["FPS", 					["Empty scene, with 1 Fetch, fluids", "Empty scene, with 1 Fetch, cloth", "Empty scene, with 1 Fetch, macro particles", "Empty scene, with 1 Fetch, cloth, fluids, macro particles"]],
	'np_loading_canvas': 			["Loading time",   			["Empty scene, with 1 Fetch, fluids", "Empty scene, with 1 Fetch, cloth", "Empty scene, with 1 Fetch, macro particles", "Empty scene, with 1 Fetch, cloth, fluids, macro particles"]],
	'np_omni_canvas': 				["Omni step time", 			["Empty scene, with 1 Fetch, fluids", "Empty scene, with 1 Fetch, cloth", "Empty scene, with 1 Fetch, macro particles", "Empty scene, with 1 Fetch, cloth, fluids, macro particles"]],
	'np_non_omni_canvas': 			["Non-omni step time", 		["Empty scene, with 1 Fetch, fluids", "Empty scene, with 1 Fetch, cloth", "Empty scene, with 1 Fetch, macro particles", "Empty scene, with 1 Fetch, cloth, fluids, macro particles"]],
	'np_mem_canvas': 				["Memory usage", 			["Empty scene, with 1 Fetch, fluids", "Empty scene, with 1 Fetch, cloth", "Empty scene, with 1 Fetch, macro particles", "Empty scene, with 1 Fetch, cloth, fluids, macro particles"]],
	'np_vram_canvas': 				["Vram usage", 				["Empty scene, with 1 Fetch, fluids", "Empty scene, with 1 Fetch, cloth", "Empty scene, with 1 Fetch, macro particles", "Empty scene, with 1 Fetch, cloth, fluids, macro particles"]],
	'scene_total_canvas': 			["FPS", 					["Ihlen_0_int, with 1 Fetch", "Pomaria_0_garden, with 1 Fetch", "house_single_floor, with 1 Fetch", "grocery_store_cafe, with 1 Fetch"]],
	'scene_loading_canvas': 		["Loading time", 			["Ihlen_0_int, with 1 Fetch", "Pomaria_0_garden, with 1 Fetch", "house_single_floor, with 1 Fetch", "grocery_store_cafe, with 1 Fetch"]],
	'scene_omni_canvas': 			["Omni step time", 			["Ihlen_0_int, with 1 Fetch", "Pomaria_0_garden, with 1 Fetch", "house_single_floor, with 1 Fetch", "grocery_store_cafe, with 1 Fetch"]],
	'scene_non_omni_canvas': 		["Non-omni step time", 		["Ihlen_0_int, with 1 Fetch", "Pomaria_0_garden, with 1 Fetch", "house_single_floor, with 1 Fetch", "grocery_store_cafe, with 1 Fetch"]],
	'scene_mem_canvas': 			["Memory usage", 			["Ihlen_0_int, with 1 Fetch", "Pomaria_0_garden, with 1 Fetch", "house_single_floor, with 1 Fetch", "grocery_store_cafe, with 1 Fetch"]],
	'scene_vram_canvas': 			["Vram usage", 				["Ihlen_0_int, with 1 Fetch", "Pomaria_0_garden, with 1 Fetch", "house_single_floor, with 1 Fetch", "grocery_store_cafe, with 1 Fetch"]],
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
				const title_map = map.get(bench.extra[0]);
				if (title_map === undefined) {
					const temp_map = new Map();
					temp_map.set(bench.name, [result]);
					map.set(bench.extra[0], temp_map);
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
		const a = document.createElement('a');
		a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}));
		a.download = 'OmniGibson Profiling.json';
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
	const color = ['#178600', '#00add8', '#ffa500', '#ff3838'];
	const data = {
		labels: Array.from(filteredData.values())[0].map(value => value.commit.id.slice(0, 7)),
		datasets: Array.from(filteredData).map(([name, value], index) => {
			return {
				label: name,
				data: value.map(d => ({
					'x': d.commit.id.slice(0, 7),
					'y': d.bench.value
				}
				)),
				borderColor: color[index],
				backgroundColor: 'rgba(0, 0, 0, 0.01)'
			};
	})
	};
	const options = {
		tooltips: {
			callbacks: {
				afterTitle: items => {
					const {datasetIndex, index} = items[0];
					const data = Array.from(filteredData.values())[datasetIndex][index];
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
					return extra ? '\n' + extra[0] : '';
				}
			}
		},
		onClick: (_mouseEvent) => {
			const points = myChart.getElementsAtEventForMode(_mouseEvent, 'nearest', { intersect: true }, true);
			if (points.length === 0) {
				return;
			}
			const url = Array.from(filteredData.values())[points[0]._datasetIndex][points[0]._index].commit.url;
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

	const myChart = new Chart(canvas, {
		type: 'line',
		data,
		options,
	});
	return myChart;
}



const allData = init()
for (const [canvasName, [fieldName, runNames]] of Object.entries(canvasDict)) {
	renderGraph(canvasName, fieldName, runNames);
}

