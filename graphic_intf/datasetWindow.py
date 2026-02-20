from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea
)
from PyQt6.QtCore import Qt
import sys

class DatasetWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CCISED - datasets")
        self.resize(800, 600)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Titolo principale
        title = QLabel("CCISED - Dataset Introduction")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title.setStyleSheet("font-weight: bold; font-size: 18px;")
        main_layout.addWidget(title)

        # Testo introduttivo
        intro_text = (
            "The first dataset used is the World Development Indicators (WDI) database, "
            "published by the World Bank. It is a comprehensive collection of global development data, "
            "providing key economic, social, and environmental statistics. It includes over 1,500 indicators "
            "covering more than 200 countries and territories, with data spanning several decades. "
            "The indicators are sourced from reputable national and international agencies, ensuring "
            "high-quality, consistent, and comparable data.\n\n"
            "The World Happiness Report is a survey of the state of global happiness that ranks countries "
            "by how ‘happy’ their citizens perceive themselves to be. This dataset has been extracted from "
            "\"Models Demystified: A Practical Guide from Linear Regression to Deep Learning\" "
            "Authors Michael Clark and Seth Berry and stems from the World Happiness Report, which combines "
            "wellbeing data with surveys made directly to citizens of all countries surveyed and finds a happiness "
            "score and ranking based on six main variables:\n\n"
            "- Having someone to count on\n"
            "- Log GDP per capita\n"
            "- Healthy life expectancy\n"
            "- Freedom to make life choices\n"
            "- Generosity\n"
            "- Freedom from corruption\n\n"
            "The dataset has been preprocessed as follows:\n\n"
            "COARSE WDI DIMENSIONALITY REDUCTION:\n"
            "- Remove all WDI years before 2005\n"
            "- Remove all attributes below 80% coverage\n\n"
            "FINE(R) WDI DIMENSIONALITY REDUCTION:\n"
            "- Exploratory PCA\n\n"
            "COARSE WHR DIMENSIONALITY REDUCTION:\n"
            "- Remove duplicates with WDI\n\n"
            "DATA INTEGRATION:\n"
            "- Remove countries without WDI observations\n\n"
            "FINAL DIMENSIONALITY REDUCTION:\n"
            "- Remove low variance features\n"
            "- Correlation analysis to remove other redundant attributes\n"
            "FINAL DATA CLEANING:\n"
            "- Fill missing values\n"
            "- Z-score normalization\n\n"
        )
        description = QLabel(intro_text)
        description.setAlignment(Qt.AlignmentFlag.AlignLeft)
        description.setWordWrap(True)
        main_layout.addWidget(description)

        # Tabella con 2 colonne e 25 righe
        table = QTableWidget(24, 2)
        table.setHorizontalHeaderLabels(["Attribute", "Description / Notes"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Abilitiamo il wrapping e le righe alternate colorate
        table.setWordWrap(True)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(
            "QTableWidget {"
            "gridline-color: #cccccc;"
            "alternate-background-color: #121212;"
            "background-color: #222222;"
            "}"
        )

        # Inseriamo tutte le righe che hai fornito
        data = [
            ("Proportion of seats held by women in national parliaments (%)",
             "Women in parliaments are the percentage of parliamentary seats in a single or lower chamber held by women. Women are vastly underrepresented in decision making positions in government, although there is some evidence of recent improvement."),
            ("Compulsory education, duration (years)",
             "Duration of compulsory education is the number of years that children are legally obliged to attend school."),
            ("Inflation, GDP deflator (annual % growth)",
             "Inflation as measured by the annual growth rate of the GDP implicit deflator shows the rate of price change in the economy as a whole. The GDP implicit deflator is the ratio of GDP in current local currency to GDP in constant local currency."),
            ("Individuals using the Internet (% of population)",
             "Internet users are individuals who have used the Internet (from any location) in the last 3 months. The Internet can be used via a computer, mobile phone, personal digital assistant, games machine, digital TV etc."),
            ("Military expenditure (% of GDP)",
             "Military expenditure by country as percentage of gross domestic product"),
            ("GDP per capita, PPP (current international $)",
             "This indicator provides values for gross domestic product (GDP) expressed in current international dollars, converted by purchasing power parities (PPPs). PPPs account for the different price levels across countries and thus PPP-based comparisons of economic output are more appropriate for comparing the output of economies and the average material well-being of their inhabitants than exchange-rate based comparisons."),
            ("Industry (including construction), value added per worker (constant 2015 US$)",
             "Industry (including construction) corresponds to ISIC (Rev.4) divisions 05-43. It is comprised of mining, manufacturing, construction, electricity, water, and gas industries. Value added is the contribution to the economy by a producer or an industry or an institutional sector, which is estimated by the total value of output produced and deducting the total value of intermediate consumption of goods and services used to produce that output. The core indicator has been divided by the number of workers in the economy to derive a measure of labor productivity. This indicator is expressed in constant prices, meaning the series has been adjusted to account for price changes over time. The reference year for this adjustment is 2015. This indicator is expressed in United States dollars."),
            ("Urban population (% of total population)",
             "Urban population refers to people living in urban areas as defined by national statistical offices. The data are collected and smoothed by United Nations Population Division."),
            ("Access to electricity (% of population)",
             "Access to electricity is the percentage of population with access to electricity. Electrification data are collected from industry, national surveys and international sources."),
            ("Renewable energy consumption (% of total final energy consumption)",
             "Renewable energy consumption is the share of renewables energy in total final energy consumption."),
            ("Land area (sq. km)",
             "Land area is a country's total area, excluding area under inland water bodies, national claims to continental shelf, and exclusive economic zones. In most cases the definition of inland water bodies includes major rivers and lakes."),
            ("Population ages 15-64 (% of total population)",
             "Total population between the ages 15 to 64 as a percentage of the total population. Population is based on the de facto definition of population, which counts all residents regardless of legal status or citizenship."),
            ("Merchandise exports by the reporting economy (current US$)",
             "Merchandise exports by the reporting economy are the total merchandise exports by the reporting economy to the rest of the world, as reported in the IMF's Direction of trade database. Data are in current US$."),
            ("Net migration",
             "Net migration is the net total of migrants during the period, that is, the number of immigrants minus the number of emigrants, including both citizens and noncitizens."),
            ("Vulnerable employment, total (% of total employment) (modeled ILO estimate)",
             "Vulnerable employment is contributing family workers and own-account workers as a percentage of total employment."),
            ("Employment to population ratio, 15+, total (%) (modeled ILO estimate)",
             "Employment to population ratio is the proportion of a country's population that is employed."),
            ("happiness_score", "The average life evaluation score for the year in question."),
            ("social_support", "If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?"),
            ("healthy_life_expectancy_at_birth", "Based on data from the World Health Organization Global Health Observatory."),
            ("freedom_to_make_life_choices", "Are you satisfied or dissatisfied with your freedom to choose what you do with your life?"),
            ("generosity", "Have you donated money to a charity in the past month?"),
            ("positive_affect", "The national average of binary responses (0=no, 1=yes) about three emotions experienced on the previous day: laughter, enjoyment, and interest."),
            ("negative_affect", "The national average of binary responses (0=no, 1=yes) about three emotions experienced on the previous day: worry, sadness, and anger."),
            ("perceptions_of_corruption", "The average of two questions: “Is corruption widespread throughout the government or not?” and “Is corruption widespread within businesses or not?” Where data for government corruption are missing, the perception of business corruption is used as the overall corruption-perception measure.")
        ]

        for row, (attr, desc) in enumerate(data):
            table.setItem(row, 0, QTableWidgetItem(attr))
            table.setItem(row, 1, QTableWidgetItem(desc))

        # Ridimensiona automaticamente le righe in base al contenuto
        table.resizeRowsToContents()

        main_layout.addWidget(table)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(main_layout)
        scroll_area.setWidget(container)

        self.setCentralWidget(scroll_area)

