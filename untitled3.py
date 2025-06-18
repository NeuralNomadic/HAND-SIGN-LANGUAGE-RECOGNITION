from graphviz import Digraph

# Initialize the flowchart
flowchart = Digraph(format="png", graph_attr={'rankdir': 'TB'}, node_attr={'shape': 'rectangle'})

# Add nodes for each step in the process
flowchart.node("Start", "Start: 50 Participants")
flowchart.node("Round1", "Round 1: 25 Matches\nWinners: 25 → 13 (1 bye)")
flowchart.node("Round2", "Round 2: 13 Matches\nWinners: 13 → 7 (1 bye)")
flowchart.node("Round3", "Round 3: 7 Matches\nWinners: 7 → 4")
flowchart.node("Semifinal", "Semifinal Round\nWinners: 4 → 2")
flowchart.node("Final", "Final Round\nWinner: 1")

# Add edges to connect nodes
flowchart.edges([
    ("Start", "Round1"),
    ("Round1", "Round2"),
    ("Round2", "Round3"),
    ("Round3", "Semifinal"),
    ("Semifinal", "Final")
])

# Render and display the flowchart
file_path = "C:/"
flowchart.render(file_path, format="png", cleanup=True)
file_path + ".png"
