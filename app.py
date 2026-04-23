from pathlib import Path

import streamlit as st


st.set_page_config(
	page_title="⛏️ Data Mining",
	page_icon="💎",
	layout="wide"
)

st.markdown(
	"""
	<style>
	.stApp {
		background: radial-gradient(ellipse at center, #000000 0%, #111111 40%, #333333 100%) !important;
		background-attachment: fixed !important;
	}
	</style>
	<canvas id="star-canvas"></canvas>
	<script>
	(function() {
		const existing = document.getElementById('star-canvas');
		if (existing && existing._initialized) return;

		const canvas = existing || document.createElement('canvas');
		canvas.id = 'star-canvas';
		canvas._initialized = true;
		canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;';
		if (!existing) document.body.appendChild(canvas);

		const ctx = canvas.getContext('2d');

		function resize() {
			canvas.width = window.innerWidth;
			canvas.height = window.innerHeight;
		}
		resize();
		window.addEventListener('resize', resize);

		const stars = Array.from({length: 250}, () => ({
			x: Math.random(),
			y: Math.random(),
			r: Math.random() * 1.2 + 0.2,
			alpha: Math.random(),
			delta: (Math.random() * 0.004 + 0.001) * (Math.random() < 0.5 ? 1 : -1)
		}));

		function draw() {
			ctx.clearRect(0, 0, canvas.width, canvas.height);
			stars.forEach(s => {
				s.alpha += s.delta;
				if (s.alpha >= 1) { s.alpha = 1; s.delta *= -1; }
				if (s.alpha <= 0.1) { s.alpha = 0.1; s.delta *= -1; }
				ctx.beginPath();
				ctx.arc(s.x * canvas.width, s.y * canvas.height, s.r, 0, Math.PI * 2);
				ctx.fillStyle = 'rgba(255,255,255,' + s.alpha + ')';
				ctx.fill();
			});
			requestAnimationFrame(draw);
		}
		draw();
	})();
	</script>
	""",
	unsafe_allow_html=True,
)


def home_page() -> None:
	st.title("Data Mining")
	st.write(
		"Choose a module tile below to open the corresponding page."
	)

	def render_tile(
		title: str,
		description: str,
		page_path: str,
		image_path: str | None,
		placeholder_text: str,
		notebook_label: str,
		notebook_link: str | None,
		key_prefix: str,
	) -> None:
		with st.container(border=True):
			if image_path and Path(image_path).exists():
				st.image(image_path, width="stretch")
			else:
				st.info(placeholder_text)

			st.markdown(f"**{title}**")

			if st.button(
				"Open page",
				use_container_width=True,
				icon=":material/open_in_new:",
				key=f"{key_prefix}_open_page",
			):
				st.switch_page(page_path)

			if notebook_link:
				st.link_button(
					notebook_label,
					notebook_link,
					use_container_width=True,
					icon=":material/menu_book:",
				)
			else:
				st.button(
					notebook_label,
					disabled=True,
					use_container_width=True,
					icon=":material/disabled_by_default:",
					key=f"{key_prefix}_notebook_disabled",
				)

	tile_cols = st.columns(4, gap="large")

	with tile_cols[0]:
		render_tile(
			title="01: Introduction to Data Mining",
			description="Overview of CRISP-DM phases and foundational concepts for the course.",
			page_path="01_introduction_to_data_mining/module_01_app.py",
			image_path="01_introduction_to_data_mining/images/01_cover.svg",
			placeholder_text="Module preview",
			notebook_label="Notebook NA",
			notebook_link=None,
			key_prefix="module_01",
		)

	with tile_cols[1]:
		render_tile(
			title="02: Recap Outlier Removal",
			description="Interactive explorer to detect potential outliers using Z-score, IQR, or domain rules.",
			page_path="02_recap_outlier_removal/module_02_app.py",
			image_path="02_recap_outlier_removal/images/02_cover.svg",
			placeholder_text="Module preview",
			notebook_label="Notebook Example",
			notebook_link="https://erickoziel.com",
			key_prefix="module_02",
		)


home = st.Page(home_page, title="Home", icon=":material/home:", default=True)
module_01 = st.Page(
	"01_introduction_to_data_mining/module_01_app.py",
	title="Module 01: Introduction to Data Mining",
	icon=":material/looks_one:"
)
module_02 = st.Page(
	"02_recap_outlier_removal/module_02_app.py",
	title="Module 02: Recap and Outlier Removal",
	icon=":material/looks_two:"
)

navigation = st.navigation(
	{
		"Course": [home],
		"Modules": [module_01, module_02],
	}
)

navigation.run()
