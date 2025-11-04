import streamlit as st
import pandas as pd
import re
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os, json, time
from pathlib import Path


TRAIN_LOG = Path("data/train_pairs_v2.jsonl")

def append_train_pair(text1, text2, label, meta=None):
    """
    Append one labeled pair to JSONL.
    label: 1 for positive (match), 0 for negative (non-match).
    meta: optional dict (schema, timestamp, cleaned variants, etc.)
    """
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "text1": str(text1 or ""),
        "text2": str(text2 or ""),
        "label": int(label),
        "ts": int(time.time())
    }
    if isinstance(meta, dict):
        rec.update(meta)
    with TRAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

@st.cache_resource(show_spinner="Loading semantic model...")
def get_sbert_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("models/sbert-finetuned-v2")

def main():
    st.title("Match Common Concept to Data")

    # --- Always ensure the model exists, then cache it in state ---
    model = st.session_state.get("sbert_model")
    if model is None:
        try:
            model = get_sbert_model()
            st.session_state["sbert_model"] = model
        except Exception as e:
            st.error(f"Could not load SBERT model: {e}")
            st.stop()

    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text or "")
        text = text.lower()
        text = text.replace("_", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep letters/numbers/spaces
        text = " ".join(text.split())  # collapse multiple spaces
        return text


    # Load user-uploaded data
    if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
        st.warning("No uploaded data found. Please start from 'Investigate Raw Data' and upload a file.")
        return

    df = st.session_state["raw_df"]
    found_dict = st.session_state.get("found_schemas", {})
    schema_rules = st.session_state.get("schema_rules", {})
    schema_attribute_rules = st.session_state.get("schema_attribute_rules", {})

    # Unique concepts, persisted for export
    found_schemas = list(dict.fromkeys([found_dict[i] for i in sorted(found_dict)]))
    st.session_state["unique_concepts"] = found_schemas

    if not found_schemas:
        st.warning("No schema mappings found. Please complete Step 2 before proceeding.")
        return

    # Select concept (schema)
    selected_schema = st.selectbox(
        "Select the concept to map attributes for:",
        found_schemas
    )

    # Apply filtering rule (if defined) to original dataframe
    try:
        query = schema_rules.get(selected_schema, "")
        filtered_df = df.query(query) if query else df
    except Exception as e:
        st.warning(f"Query failed: {e}")
        filtered_df = df
        # Keep all columns, even if all values are None
    cleaned_df = filtered_df

    # cleaned_df = filtered_df.dropna(axis=1, how='all')

    # 1) Get attributes from Show_Common_Concept.py
    concept_attributes = st.session_state.get("system_unit_class_attributes", {}).get(selected_schema, [])
    if not concept_attributes:
        st.warning(
            f"No attributes found for '{selected_schema}'. Please go to the 'Common Concept' app and select this node."
        )
        return

    # 2) Show filtered data table (always shown so you can see what you're mapping against)
    with st.expander("Filtered Data table", expanded=False):
        st.dataframe(cleaned_df, use_container_width=True)

    # 3) Toggle mapping direction
    st.divider()
    map_data_to_concept = st.checkbox("Map Data → Concept (instead of Concept → Data)", value=False)

    # Prepare common bits
    data_attribute_list = list(dict.fromkeys(cleaned_df.columns.tolist()))
    saved_attr_mapping = st.session_state.get("schema_attribute_rules", {}).get(selected_schema, {})
    saved_type_mapping = st.session_state.get("schema_attribute_types", {}).get(selected_schema, {})

    # --- NEW: build cleaned variants for embeddings/similarity
    domain_columns_original = data_attribute_list
    domain_columns_clean = [preprocess_text(c) for c in domain_columns_original]

    # Persist for later (SBERT & auto-suggest)
    st.session_state["domain_columns_original"] = domain_columns_original
    st.session_state["domain_columns_clean"] = domain_columns_clean


    if not map_data_to_concept:
        st.subheader("Map Concept Attributes → Data Columns")

        # --- SBERT Embedding + Similarity step ---
        # Clean the attributes actually shown in the table
        concept_attributes_clean = [preprocess_text(a) for a in concept_attributes]
        domain_clean = st.session_state.get("domain_columns_clean", [])
        domain_original = st.session_state.get("domain_columns_original", [])

        if concept_attributes_clean and domain_clean:
            # Encode CURRENT concept attributes (not a stale aml_clean)
            aml_embeddings = model.encode(concept_attributes_clean)
            domain_embeddings = model.encode(domain_clean)

            sim_matrix = cosine_similarity(aml_embeddings, domain_embeddings)
            best_match_indices = np.argmax(sim_matrix, axis=1)
            best_matches = [domain_original[i] for i in best_match_indices]
            best_scores = [sim_matrix[row, col] for row, col in enumerate(best_match_indices)]

            # Store session keys in the SAME order as the shown table (for logging later)
            st.session_state["auto_suggested_matches"] = best_matches
            st.session_state["auto_suggested_scores"] = best_scores
            st.session_state["sim_matrix"] = sim_matrix
            st.session_state["aml_attr_list"] = concept_attributes  # originals in the table
            st.session_state["aml_attr_clean"] = concept_attributes_clean  # cleaned in the same order
            st.session_state["domain_cols_list"] = domain_original

            # --- Apply threshold-based auto-fill (now 1:1 with concept_attributes) ---
            threshold = 0.35
            data_attr_values = []
            for idx, attr in enumerate(concept_attributes):
                saved_value = saved_attr_mapping.get(attr, "")
                if not saved_value:
                    saved_value = best_matches[idx] if best_scores[idx] >= threshold else ""
                data_attr_values.append(saved_value)

            # Build initial DataFrame for editor (Similarity has no None now)
            mapped_df = pd.DataFrame({
                "Concept Data": concept_attributes,
                "Data Attr": data_attr_values,
                "Type": [saved_type_mapping.get(attr, "string") for attr in concept_attributes],
                "Similarity": [round(s, 3) for s in best_scores]
            })
        else:
            # Fallback if we don't have domain columns yet
            mapped_df = pd.DataFrame({
                "Concept Data": concept_attributes,
                "Data Attr": [saved_attr_mapping.get(attr, "") for attr in concept_attributes],
                "Type": [saved_type_mapping.get(attr, "string") for attr in concept_attributes],
                "Similarity": [None] * len(concept_attributes)
            })

        # Optional quick debug
        st.caption(f"len(concept_attributes)={len(concept_attributes)} | "
                   f"len(sim)={'0' if 'best_scores' not in locals() else len(best_scores)}")

        # Keep old mapping for comparison
        old_mapping = saved_attr_mapping.copy()

        mapped_df = st.data_editor(
            mapped_df,
            use_container_width=True,
            key=f"editor_forward_{selected_schema}",
            column_config={
                "Data Attr": st.column_config.SelectboxColumn(
                    label="Choose the appropriate mapping",
                    options=data_attribute_list,
                ),
                "Type": st.column_config.SelectboxColumn(
                    label="Select data type",
                    options=["string", "int", "float", "boolean"],
                )
            },
            disabled=["Concept Data"]
        )

        # Collect success messages for changed mappings
        success_msgs = []
        for concept_attr, new_data_attr in zip(mapped_df["Concept Data"], mapped_df["Data Attr"]):
            if new_data_attr and old_mapping.get(concept_attr) != new_data_attr:
                success_msgs.append(f"'{concept_attr}' → '{new_data_attr}'")

        # Display in columns (3 per row)
        if success_msgs:
            num_cols = 5
            rows = [success_msgs[i:i + num_cols] for i in range(0, len(success_msgs), num_cols)]
            for row in rows:
                cols = st.columns(len(row))
                for col, msg in zip(cols, row):
                    col.success(msg)

        if st.button("✅ Validate Mapping", use_container_width=True, type="primary",
                     key=f"validate_forward_{selected_schema}"):

            attr_mapping = dict(zip(mapped_df["Concept Data"], mapped_df["Data Attr"]))
            type_mapping = dict(zip(mapped_df["Concept Data"], mapped_df["Type"]))

            # Save to session (your existing code)
            st.session_state.setdefault("schema_attribute_rules", {})[selected_schema] = attr_mapping
            st.session_state.setdefault("schema_attribute_types", {})[selected_schema] = type_mapping
            st.success("Mapping saved (Concept → Data).")

            # --- NEW: Log positives + K hard negatives per attribute ---
            K = 3  # how many hard negatives to log
            sim = st.session_state.get("sim_matrix")
            aml_list = st.session_state.get("aml_attr_list", [])
            dom_list = st.session_state.get("domain_cols_list", [])
            aml_clean = st.session_state.get("aml_attr_clean", [])
            dom_clean = st.session_state.get("domain_columns_clean", [])

            # Build a quick index to row map for sim_matrix
            idx_by_attr = {a: i for i, a in enumerate(aml_list)} if aml_list else {}
            auto_suggestions = st.session_state.get("auto_suggested_matches", [])
            auto_scores = st.session_state.get("auto_suggested_scores", [])
            logged_negs = set()  # to avoid duplicate negatives

            for concept_attr, chosen_col in attr_mapping.items():
                # row index for this concept (used across all branches)
                r = idx_by_attr.get(concept_attr, None)

                # model’s original suggestion for this row (if any)
                suggested_col = auto_suggestions[r] if (r is not None and r < len(auto_suggestions)) else None
                suggested_score = auto_scores[r] if (r is not None and r < len(auto_scores)) else None

                # ---------- CASE A: user cleared the cell (None/empty) ----------
                if not chosen_col:
                    # If there was a suggestion, log it as user-rejected negative
                    if suggested_col:
                        meta_reject = {
                            "schema": selected_schema,
                            "mode": "concept_to_data",
                            "user_rejected": True,
                            "suggested_score": float(suggested_score) if suggested_score is not None else None,
                            "text1_clean": (aml_clean[r] if (r is not None and r < len(aml_clean)) else ""),
                            "text2_clean": (
                                dom_clean[dom_list.index(suggested_col)] if suggested_col in dom_list else "")
                        }
                        append_train_pair(concept_attr, suggested_col, 0, meta_reject)
                        logged_negs.add(suggested_col)
                    continue  # nothing else to log for this attribute

                # ---------- CASE B: POSITIVE (user’s final choice) ----------
                meta_pos = {
                    "schema": selected_schema,
                    "mode": "concept_to_data",
                    "text1_clean": (aml_clean[r] if (r is not None and r < len(aml_clean)) else ""),
                    "text2_clean": (dom_clean[dom_list.index(chosen_col)] if chosen_col in dom_list else "")
                }
                append_train_pair(concept_attr, chosen_col, 1, meta_pos)

                # ---------- CASE C: corrected negative (model suggested X, user chose Y) ----------
                if suggested_col and suggested_col != chosen_col and suggested_col not in logged_negs:
                    meta_neg_corr = {
                        "schema": selected_schema,
                        "mode": "concept_to_data",
                        "corrected_negative": True,
                        "suggested_score": float(suggested_score) if suggested_score is not None else None,
                        "text1_clean": (aml_clean[r] if (r is not None and r < len(aml_clean)) else ""),
                        "text2_clean": (dom_clean[dom_list.index(suggested_col)] if suggested_col in dom_list else "")
                    }
                    append_train_pair(concept_attr, suggested_col, 0, meta_neg_corr)
                    logged_negs.add(suggested_col)

                # ---------- CASE D: hard negatives (top-K by similarity) ----------
                if sim is not None and r is not None and r < sim.shape[0]:
                    cand_idx = sim[r].argsort()[::-1]  # descending
                    negs = []
                    for j in cand_idx:
                        if j >= len(dom_list):
                            continue
                        col_name = dom_list[j]
                        if col_name != chosen_col and col_name not in logged_negs:
                            negs.append((j, col_name))
                        if len(negs) == K:
                            break

                    for j, col_name in negs:
                        meta_neg = {
                            "schema": selected_schema,
                            "mode": "concept_to_data",
                            "text1_clean": (aml_clean[r] if (r is not None and r < len(aml_clean)) else ""),
                            "text2_clean": (dom_clean[j] if j < len(dom_clean) else "")
                        }
                        append_train_pair(concept_attr, col_name, 0, meta_neg)
                        logged_negs.add(col_name)

    else:

        # ---------- Data → Concept ----------
        st.subheader("Map Data Columns → Concept Attributes")
        # --- Build reverse defaults from saved mapping (if any) ---
        reverse_saved = {}
        for c_attr, col in saved_attr_mapping.items():
            if col:
                reverse_saved[col] = c_attr

        # Reverse type mapping: column -> type
        reverse_types = {}

        for c_attr, col in saved_attr_mapping.items():
            if col:
                reverse_types[col] = saved_type_mapping.get(c_attr, "string")

        # --- SBERT Embedding + Similarity (rows = data cols, cols = concept attrs) ---
        concept_clean = [preprocess_text(a) for a in concept_attributes]
        data_cols_clean = [preprocess_text(c) for c in data_attribute_list]
        if data_attribute_list and concept_attributes:
            data_emb = model.encode(data_cols_clean)
            concept_emb = model.encode(concept_clean)
            # shape: (n_data_cols, n_concepts)
            sim_matrix_rev = cosine_similarity(data_emb, concept_emb)
            # best concept per data column
            best_idx_rev = np.argmax(sim_matrix_rev, axis=1)
            best_concepts = [concept_attributes[j] for j in best_idx_rev]
            best_scores_rev = [sim_matrix_rev[i, j] for i, j in enumerate(best_idx_rev)]

            # Store session keys (aligned to data_attribute_list order)
            st.session_state["sim_matrix_rev"] = sim_matrix_rev
            st.session_state["data_cols_list"] = data_attribute_list
            st.session_state["data_cols_clean"] = data_cols_clean
            st.session_state["concept_list"] = concept_attributes
            st.session_state["concept_clean"] = concept_clean
            st.session_state["auto_suggested_concepts"] = best_concepts
            st.session_state["auto_suggested_scores_r"] = best_scores_rev

            # --- Prefill Concept column only if score >= threshold ---

            threshold = 0.35
            concept_prefill = []
            for i, col in enumerate(data_attribute_list):
                saved = reverse_saved.get(col, "")

                if not saved:
                    saved = best_concepts[i] if best_scores_rev[i] >= threshold else ""

                concept_prefill.append(saved)

            mapped_df_rev = pd.DataFrame({
                "Data Attr": data_attribute_list,
                "Concept Data": concept_prefill,
                "Type": [reverse_types.get(col, "string") for col in data_attribute_list],
                "Similarity": [round(s, 3) for s in best_scores_rev]
            })

        else:
            # Fallback (no embedding yet)
            mapped_df_rev = pd.DataFrame({
                "Data Attr": data_attribute_list,
                "Concept Data": [reverse_saved.get(col, "") for col in data_attribute_list],
                "Type": [reverse_types.get(col, "string") for col in data_attribute_list],
                "Similarity": [None] * len(data_attribute_list),
            })
        # Keep old mapping for comparison
        old_reverse_mapping = reverse_saved.copy()
        mapped_df_rev = st.data_editor(
            mapped_df_rev,
            use_container_width=True,
            key=f"editor_reverse_{selected_schema}",
            column_config={
                "Concept Data": st.column_config.SelectboxColumn(
                    label="Choose Concept Attribute",
                    options=concept_attributes,
                ),
                "Type": st.column_config.SelectboxColumn(
                    label="Select data type",
                    options=["string", "int", "float", "boolean"],
                ),
                "Similarity": st.column_config.NumberColumn(
                    label="Similarity",
                    format="%.03f",
                    disabled=True
                ),
            },
            disabled=["Data Attr"]
        )

        # Collect success messages for changed mappings
        success_msgs = []
        for data_col, new_concept_attr in zip(mapped_df_rev["Data Attr"], mapped_df_rev["Concept Data"]):
            if new_concept_attr and old_reverse_mapping.get(data_col) != new_concept_attr:
                success_msgs.append(f"'{data_col}' → '{new_concept_attr}'")

        if success_msgs:
            num_cols = 5
            rows = [success_msgs[i:i + num_cols] for i in range(0, len(success_msgs), num_cols)]
            for row in rows:
                cols = st.columns(len(row))
                for col, msg in zip(cols, row):
                    col.success(msg)

        if st.button("✅ Validate Mapping", use_container_width=True, type="primary",
                     key=f"validate_reverse_{selected_schema}"):
            # ---- Build forward map from reverse table (your existing behavior) ----
            forward_map = {}
            type_mapping = {}
            for _, row in mapped_df_rev.iterrows():
                concept_attr = (row.get("Concept Data") or "").strip()
                data_col = row.get("Data Attr")
                if concept_attr:
                    forward_map[concept_attr] = data_col
                    type_mapping[concept_attr] = row.get("Type", "string")
            st.session_state.setdefault("schema_attribute_rules", {})[selected_schema] = forward_map
            st.session_state.setdefault("schema_attribute_types", {})[selected_schema] = type_mapping
            st.success("Mapping saved (Data → Concept).")

            # ---- Logging: positives, corrected negatives, user rejections, hard negatives ----

            # Use the reverse sim matrix where rows = data columns
            simr = st.session_state.get("sim_matrix_rev")
            data_cols_list = st.session_state.get("data_cols_list", [])
            concept_list = st.session_state.get("concept_list", [])
            data_clean_list = st.session_state.get("data_cols_clean", [])
            concept_clean_l = st.session_state.get("concept_clean", [])
            auto_concepts = st.session_state.get("auto_suggested_concepts", [])
            auto_scores_r = st.session_state.get("auto_suggested_scores_r", [])
            # Fast lookups

            idx_by_data = {name: i for i, name in enumerate(data_cols_list)} if data_cols_list else {}
            K = 3
            logged_negs = set()
            for _, row in mapped_df_rev.iterrows():
                data_col = row.get("Data Attr")
                chosen_concept = (row.get("Concept Data") or "").strip()
                i = idx_by_data.get(data_col, None)
                suggested_concept = auto_concepts[i] if (i is not None and i < len(auto_concepts)) else None
                suggested_score = auto_scores_r[i] if (i is not None and i < len(auto_scores_r)) else None

                # CASE A: user cleared (no concept selected) -> log suggestion as user_rejected
                if not chosen_concept:
                    if suggested_concept:
                        meta_reject = {
                            "schema": selected_schema,
                            "mode": "data_to_concept",
                            "user_rejected": True,
                            "suggested_score": float(suggested_score) if suggested_score is not None else None,
                            "text1_clean": (concept_clean_l[concept_list.index(suggested_concept)]
                                            if suggested_concept in concept_list else ""),
                            "text2_clean": (data_clean_list[i] if i is not None and i < len(data_clean_list) else "")
                        }

                        # Training pair is (Concept, Data)

                        append_train_pair(suggested_concept, data_col, 0, meta_reject)
                        logged_negs.add((suggested_concept, data_col))
                    continue

                # CASE B: positive mapping (user’s final choice)

                meta_pos = {
                    "schema": selected_schema,
                    "mode": "data_to_concept",
                    "text1_clean": (concept_clean_l[concept_list.index(chosen_concept)]
                                    if chosen_concept in concept_list else ""),
                    "text2_clean": (data_clean_list[i] if i is not None and i < len(data_clean_list) else "")
                }
                append_train_pair(chosen_concept, data_col, 1, meta_pos)

                # CASE C: corrected negative (model suggested X, user chose Y)
                if suggested_concept and suggested_concept != chosen_concept:
                    key_neg = (suggested_concept, data_col)
                    if key_neg not in logged_negs:
                        meta_neg_corr = {
                            "schema": selected_schema,
                            "mode": "data_to_concept",
                            "corrected_negative": True,
                            "suggested_score": float(suggested_score) if suggested_score is not None else None,
                            "text1_clean": (concept_clean_l[concept_list.index(suggested_concept)]
                                            if suggested_concept in concept_list else ""),
                            "text2_clean": (data_clean_list[i] if i is not None and i < len(data_clean_list) else "")
                        }
                        append_train_pair(suggested_concept, data_col, 0, meta_neg_corr)
                        logged_negs.add(key_neg)

                # CASE D: hard negatives (top-K alternate concepts for this data column)

                if simr is not None and i is not None and i < simr.shape[0]:
                    cand_idx = simr[i].argsort()[::-1]  # best → worst
                    negs = []
                    for j in cand_idx:
                        if j >= len(concept_list):
                            continue
                        c_name = concept_list[j]
                        if c_name != chosen_concept and (c_name, data_col) not in logged_negs:
                            negs.append((j, c_name))
                        if len(negs) == K:
                            break

                    for j, c_name in negs:
                        meta_neg = {
                            "schema": selected_schema,
                            "mode": "data_to_concept",
                            "text1_clean": (concept_clean_l[j] if j < len(concept_clean_l) else ""),
                            "text2_clean": (data_clean_list[i] if i is not None and i < len(data_clean_list) else "")
                        }

                        append_train_pair(c_name, data_col, 0, meta_neg)
                        logged_negs.add((c_name, data_col))
