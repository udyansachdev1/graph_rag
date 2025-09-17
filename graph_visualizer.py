import pickle
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
import random
from collections import defaultdict

class GraphVisualizer:
    def __init__(self, pkl_path="/workspaces/graph_rag/kg.pkl"):
        self.pkl_path = pkl_path
        self.graph = None
        self.entities = {}
        self.relationships = []
        self.load_data()
    
    def load_data(self):
        """Load the knowledge graph data from pickle file"""
        try:
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct NetworkX graph
            if 'graph' in data:
                self.graph = nx.node_link_graph(data['graph'])
            else:
                self.graph = nx.Graph()
            
            # Load entities and relationships
            self.entities = data.get('entities', {})
            self.relationships = data.get('relationships', [])
            
            print(f"✅ Loaded graph with {len(self.entities)} entities and {len(self.relationships)} relationships")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def matplotlib_visualization(self, save_path="/workspaces/graph_rag/graph_matplotlib.png"):
        """Create a static visualization using matplotlib"""
        if not self.graph:
            print("❌ No graph data loaded")
            return
        
        plt.figure(figsize=(15, 12))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Color nodes by entity type
        entity_types = {}
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
        for node in self.graph.nodes():
            if node in self.entities:
                entity_type = self.entities[node].get('type', 'Unknown')
                if entity_type not in entity_types:
                    entity_types[entity_type] = colors[len(entity_types) % len(colors)]
        
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            if node in self.entities:
                entity_type = self.entities[node].get('type', 'Unknown')
                node_colors.append(entity_types.get(entity_type, '#CCCCCC'))
                # Size based on degree (number of connections)
                node_sizes.append(300 + self.graph.degree(node) * 100)
            else:
                node_colors.append('#CCCCCC')
                node_sizes.append(300)
        
        # Draw the graph
        nx.draw(self.graph, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7,
                width=2)
        
        # Add legend
        legend_elements = []
        for entity_type, color in entity_types.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, label=entity_type))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("Knowledge Graph Visualization", size=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Static visualization saved to: {save_path}")
        plt.show()
    
    def plotly_interactive_visualization(self, save_path="/workspaces/graph_rag/graph_interactive.html"):
        """Create an interactive visualization using Plotly"""
        if not self.graph:
            print("❌ No graph data loaded")
            return
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50, dim=2)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        # Color mapping for entity types
        entity_types = set()
        for node in self.graph.nodes():
            if node in self.entities:
                entity_types.add(self.entities[node].get('type', 'Unknown'))
        
        color_map = {et: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)] 
                    for i, et in enumerate(entity_types)}
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node in self.entities:
                entity = self.entities[node]
                entity_type = entity.get('type', 'Unknown')
                description = entity.get('description', 'No description')[:100]
                
                hover_text = f"<b>{node}</b><br>"
                hover_text += f"Type: {entity_type}<br>"
                hover_text += f"Description: {description}..."
                hover_text += f"<br>Connections: {self.graph.degree(node)}"
                
                node_text.append(hover_text)
                node_colors.append(color_map.get(entity_type, '#CCCCCC'))
                node_sizes.append(20 + self.graph.degree(node) * 5)
            else:
                node_text.append(f"<b>{node}</b><br>Connections: {self.graph.degree(node)}")
                node_colors.append('#CCCCCC')
                node_sizes.append(20)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Find relationship info
            rel_info = "Connected"
            for rel in self.relationships:
                if (rel['source'] == edge[0] and rel['target'] == edge[1]) or \
                   (rel['source'] == edge[1] and rel['target'] == edge[0]):
                    rel_info = f"{rel['relationship']}: {rel['description'][:50]}..."
                    break
            edge_info.append(rel_info)
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=2, color='rgba(125,125,125,0.5)'),
                                hoverinfo='none',
                                mode='lines',
                                name='Relationships'))
        
        # Add nodes
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                marker=dict(size=node_sizes,
                                          color=node_colors,
                                          line=dict(width=2, color='white')),
                                text=[node for node in self.graph.nodes()],
                                textposition="middle center",
                                textfont=dict(size=10, color="white"),
                                hovertext=node_text,
                                hoverinfo='text',
                                name='Entities'))
        
        # Update layout
        fig.update_layout(
            title=dict(text="Interactive Knowledge Graph", x=0.5, font=dict(size=20)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes for details. Zoom and pan to explore!",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=800
        )
        
        # Save interactive plot
        fig.write_html(save_path)
        print(f"🌐 Interactive visualization saved to: {save_path}")
        print(f"   Open this file in your browser to explore the graph!")
        
        # Show in Codespaces (if possible)
        fig.show()
    
    def create_network_statistics(self):
        """Generate network statistics and visualizations"""
        if not self.graph:
            print("❌ No graph data loaded")
            return
        
        print("\n📈 NETWORK STATISTICS")
        print("=" * 50)
        
        # Basic stats
        print(f"🔢 Basic Metrics:")
        print(f"   - Nodes (Entities): {self.graph.number_of_nodes()}")
        print(f"   - Edges (Relationships): {self.graph.number_of_edges()}")
        print(f"   - Average Degree: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.2f}")
        
        # Centrality measures
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            print(f"\n🎯 Most Important Entities (by centrality):")
            
            # Top entities by degree centrality
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   By Connections (Degree Centrality):")
            for entity, score in top_degree:
                entity_type = self.entities.get(entity, {}).get('type', 'Unknown')
                print(f"     - {entity} ({entity_type}): {score:.3f}")
            
            # Top entities by betweenness centrality
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   By Bridge Importance (Betweenness Centrality):")
            for entity, score in top_betweenness:
                entity_type = self.entities.get(entity, {}).get('type', 'Unknown')
                print(f"     - {entity} ({entity_type}): {score:.3f}")
            
        except Exception as e:
            print(f"   ⚠️  Could not calculate centrality measures: {e}")
        
        # Entity type distribution
        type_counts = defaultdict(int)
        for entity_name, entity_data in self.entities.items():
            entity_type = entity_data.get('type', 'Unknown')
            type_counts[entity_type] += 1
        
        print(f"\n🏷️  Entity Types Distribution:")
        for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {entity_type}: {count}")
        
        # Relationship types
        rel_types = defaultdict(int)
        for rel in self.relationships:
            rel_types[rel.get('relationship', 'Unknown')] += 1
        
        print(f"\n🔗 Relationship Types:")
        for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {rel_type}: {count}")
    
    def create_entity_type_chart(self, save_path="/workspaces/graph_rag/entity_types.html"):
        """Create a pie chart of entity types"""
        type_counts = defaultdict(int)
        for entity_name, entity_data in self.entities.items():
            entity_type = entity_data.get('type', 'Unknown')
            type_counts[entity_type] += 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Distribution of Entity Types",
            showlegend=True,
            width=800,
            height=600
        )
        
        fig.write_html(save_path)
        print(f"📊 Entity type chart saved to: {save_path}")
        fig.show()

def main():
    """Main function to create all visualizations"""
    print("🎨 Creating Knowledge Graph Visualizations...")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = GraphVisualizer()
    
    if visualizer.graph is None:
        print("❌ Could not load graph data. Please check your PKL file.")
        return
    
    # Create static visualization
    print("\n1. Creating static matplotlib visualization...")
    visualizer.matplotlib_visualization()
    
    # Create interactive visualization
    print("\n2. Creating interactive Plotly visualization...")
    visualizer.plotly_interactive_visualization()
    
    # Create network statistics
    print("\n3. Generating network statistics...")
    visualizer.create_network_statistics()
    
    # Create entity type chart
    print("\n4. Creating entity type distribution chart...")
    visualizer.create_entity_type_chart()
    
    print("\n✅ All visualizations created!")
    print("📁 Files saved in /workspaces/graph_rag/:")
    print("   - graph_matplotlib.png (static image)")
    print("   - graph_interactive.html (interactive graph)")
    print("   - entity_types.html (entity distribution)")

if __name__ == "__main__":
    main()