import pandas as pd
import sweetviz as sv

# Load dataset Titanic
df_titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Generate report
report = sv.analyze(df_titanic)

report.show_notebook()