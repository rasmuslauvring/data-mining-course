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
	html, body, .stApp, [data-testid="stAppViewContainer"] {
		background-image:
			radial-gradient(1px 1px at  5%  8%, rgba(255,255,255,.85), transparent),
			radial-gradient(1px 1px at 12%  3%, rgba(255,255,255,.65), transparent),
			radial-gradient(1px 1px at 18% 20%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 25%  7%, rgba(255,255,255,.60), transparent),
			radial-gradient(1px 1px at 32% 15%, rgba(255,255,255,.80), transparent),
			radial-gradient(1px 1px at 40%  4%, rgba(255,255,255,.75), transparent),
			radial-gradient(1px 1px at 48% 22%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 55%  9%, rgba(255,255,255,.65), transparent),
			radial-gradient(1px 1px at 63% 18%, rgba(255,255,255,.85), transparent),
			radial-gradient(1px 1px at 70%  6%, rgba(255,255,255,.70), transparent),
			radial-gradient(1px 1px at 78% 25%, rgba(255,255,255,.80), transparent),
			radial-gradient(1px 1px at 85% 12%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 92%  4%, rgba(255,255,255,.60), transparent),
			radial-gradient(1px 1px at 97% 20%, rgba(255,255,255,.75), transparent),
			radial-gradient(1px 1px at  3% 35%, rgba(255,255,255,.80), transparent),
			radial-gradient(1px 1px at 10% 42%, rgba(255,255,255,.65), transparent),
			radial-gradient(1px 1px at 17% 38%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 24% 50%, rgba(255,255,255,.70), transparent),
			radial-gradient(1px 1px at 31% 44%, rgba(255,255,255,.85), transparent),
			radial-gradient(1px 1px at 38% 32%, rgba(255,255,255,.75), transparent),
			radial-gradient(1px 1px at 45% 55%, rgba(255,255,255,.60), transparent),
			radial-gradient(1px 1px at 52% 38%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 60% 45%, rgba(255,255,255,.70), transparent),
			radial-gradient(1px 1px at 67% 36%, rgba(255,255,255,.80), transparent),
			radial-gradient(1px 1px at 74% 52%, rgba(255,255,255,.65), transparent),
			radial-gradient(1px 1px at 82% 40%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 90% 48%, rgba(255,255,255,.75), transparent),
			radial-gradient(1px 1px at 96% 35%, rgba(255,255,255,.85), transparent),
			radial-gradient(1px 1px at  7% 60%, rgba(255,255,255,.70), transparent),
			radial-gradient(1px 1px at 14% 68%, rgba(255,255,255,.80), transparent),
			radial-gradient(1px 1px at 21% 62%, rgba(255,255,255,.65), transparent),
			radial-gradient(1px 1px at 28% 75%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 35% 65%, rgba(255,255,255,.75), transparent),
			radial-gradient(1px 1px at 43% 72%, rgba(255,255,255,.85), transparent),
			radial-gradient(1px 1px at 50% 58%, rgba(255,255,255,.60), transparent),
			radial-gradient(1px 1px at 58% 70%, rgba(255,255,255,.80), transparent),
			radial-gradient(1px 1px at 65% 63%, rgba(255,255,255,.70), transparent),
			radial-gradient(1px 1px at 72% 78%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 80% 66%, rgba(255,255,255,.65), transparent),
			radial-gradient(1px 1px at 88% 72%, rgba(255,255,255,.85), transparent),
			radial-gradient(1px 1px at 95% 60%, rgba(255,255,255,.75), transparent),
			radial-gradient(1px 1px at  2% 85%, rgba(255,255,255,.80), transparent),
			radial-gradient(1px 1px at  9% 90%, rgba(255,255,255,.70), transparent),
			radial-gradient(1px 1px at 16% 82%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 23% 95%, rgba(255,255,255,.60), transparent),
			radial-gradient(1px 1px at 30% 88%, rgba(255,255,255,.85), transparent),
			radial-gradient(1px 1px at 37% 80%, rgba(255,255,255,.75), transparent),
			radial-gradient(1px 1px at 44% 92%, rgba(255,255,255,.65), transparent),
			radial-gradient(1px 1px at 51% 85%, rgba(255,255,255,.90), transparent),
			radial-gradient(1px 1px at 58% 90%, rgba(255,255,255,.70), transparent),
			radial-gradient(1px 1px at 65% 83%, rgba(255,255,255,.80), transparent),
			radial-gradient(1px 1px at 72% 95%, rgba(255,255,255,.65), transparent),
			radial-gradient(1px 1px at 79% 87%, rgba(255,255,255,.85), transparent),
			radial-gradient(1px 1px at 86% 92%, rgba(255,255,255,.75), transparent),
			radial-gradient(1px 1px at 93% 80%, rgba(255,255,255,.90), transparent),
			radial-gradient(2px 2px at 15% 12%, rgba(255,255,255,.95), transparent),
			radial-gradient(2px 2px at 35% 28%, rgba(255,255,255,.90), transparent),
			radial-gradient(2px 2px at 55% 18%, rgba(255,255,255,.95), transparent),
			radial-gradient(2px 2px at 75%  8%, rgba(255,255,255,.90), transparent),
			radial-gradient(2px 2px at 88% 30%, rgba(255,255,255,.95), transparent),
			radial-gradient(2px 2px at 22% 55%, rgba(255,255,255,.90), transparent),
			radial-gradient(2px 2px at 45% 70%, rgba(255,255,255,.85), transparent),
			radial-gradient(2px 2px at 65% 85%, rgba(255,255,255,.95), transparent),
			radial-gradient(2px 2px at 80% 58%, rgba(255,255,255,.90), transparent),
			radial-gradient(2px 2px at 10% 75%, rgba(255,255,255,.85), transparent),
			radial-gradient(ellipse at center, #111111 100%, #111111 40%, #333333 100%) !important;
		background-attachment: fixed !important;
		background-color: #000 !important;
	}
	[data-testid="stSidebar"] {
		background-color: rgba(10, 10, 10, 0.85) !important;
	}
	</style>
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
