# Utilities API Reference

This page provides auto-generated API documentation for key utility classes and functions.

## Event Bus

::: utils.event_bus.EventBus
    options:
      show_root_heading: true
      show_source: false
      members:
        - subscribe
        - unsubscribe
        - publish

## CRDTs

::: utils.crdt.PNCounter
    options:
      show_root_heading: true
      show_source: false
      members:
        - increment
        - decrement
        - value
        - merge
        - state
        - to_dict
        - from_dict

## Monitoring

::: utils.monitoring.AgentMonitor
    options:
      show_root_heading: true
      show_source: false
      members:
        - record_metrics
        - detect_drift
        - trigger_alert
        - recommend_adaptation

## Planning

::: utils.planning.FulfillmentPlanner
    options:
      show_root_heading: true
      show_source: false
      members:
        - add_order
        - add_associate
        - plan
        - explain_plan
        - visualize_plan

::: utils.planning.StoreLayout
    options:
      show_root_heading: true
      show_source: false
      members:
        - add_obstacle
        - add_section
        - shortest_path
        - optimize_path
        - visualize

::: utils.planning.calculate_remediation_timeline
    options:
      show_root_heading: true
      show_source: false

## OpenAI Utilities

::: utils.openai_utils.safe_chat_completion
    options:
      show_root_heading: true
      show_source: false

## NLP Helpers

*(NLP helpers like `classify_intent` are generally simpler functions and might be documented sufficiently via concepts/usage examples, but could be added here if desired)*

# ::: utils.nlp.classify_intent
#     options:
#       show_root_heading: true
#       show_source: false


</rewritten_file> 