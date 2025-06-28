
# Ground Truth Annotation Guide

## Quick Annotation Tips:
1. Focus on the most important fields first: surname, first_name, home_address
2. Use annotation_hints as starting points
3. Mark parsing_difficulty: 'easy', 'medium', or 'hard'
4. Leave fields empty if not present (don't guess)

## Common Patterns:
- "Smith John, clk, r 123 Main st."
  - surname: "Smith"
  - first_name: "John" 
  - occupation: "clerk"
  - residence_indicator: "resides"
  - home_address: "123 Main st"

- "Mary, student, b same."
  - surname: [inherited from previous]
  - first_name: "Mary"
  - occupation: "student"
  - residence_indicator: "boards"
  - home_address: "same"

- "Brown Sarah (wid George), r 789 Pine st."
  - surname: "Brown"
  - first_name: "Sarah"
  - spouse_name: "George"
  - home_address: "789 Pine st"

## Annotation Priority:
1. surname (most critical)
2. first_name (most critical)  
3. home_address (high priority)
4. occupation (medium priority)
5. everything else (lower priority)

## Time-Saving Tips:
- Annotate 5-10 entries at a time, then take a break
- Start with 'easy' entries to build momentum
- Use copy-paste for repeated surnames/addresses
- Mark unclear cases as 'hard' and move on
