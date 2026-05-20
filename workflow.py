from langgraph.graph import StateGraph, END
from state import ResumeState
from nodes import (
    parse_resume_node,
    nlp_processing_node,
    vector_lookup_node,
    semantic_match_node,
    scoring_node,
    persistence_node,
)

def create_workflow():
    workflow = StateGraph(ResumeState)
    workflow.add_node("parse",          parse_resume_node)
    workflow.add_node("nlp_processing", nlp_processing_node)
    workflow.add_node("lookup",         vector_lookup_node)
    workflow.add_node("semantic_match", semantic_match_node)
    workflow.add_node("score",          scoring_node)
    workflow.add_node("persist",        persistence_node)
    workflow.set_entry_point("parse")
    workflow.add_edge("parse",          "nlp_processing")
    workflow.add_edge("nlp_processing", "lookup")
    workflow.add_edge("lookup",         "semantic_match")
    workflow.add_edge("semantic_match", "score")
    workflow.add_edge("score",          "persist")
    workflow.add_edge("persist",        END)
    return workflow.compile()