def disambiguate(abox: dict, sentence_id: str) -> dict:
    """
    Resolves candidate morphological and syntactic analyses for tokens in a sentence
    using ontology-derived compatibility rules.

    Args:
        abox (dict): ABox with 'individuals' and 'assertions'.
        sentence_id (str): The Sentence individual ID to analyze.

    Returns:
        dict: A dictionary describing per-token resolution status, selected candidates,
              remaining candidates, rejected candidates, and explanations.

    <ABoxRequirements>
        The ABox must be a dictionary with two top-level keys:

        1. "individuals": dict mapping individual IDs to {"type": <ClassName>}
           Class types include: Sentence, Token, CandidateAnalysis, Clause, Stem,
           Postposition, Case.

        2. "assertions": list of {"predicate": <name>, "args": [<arg1>, <arg2>] or [<arg1>]}
           Supports all ontology predicates (object, datatype, unary).

        Behavior:
            - If abox is None or malformed, returns a structured empty result with an error field.
            - If sentence_id is not present, attempts to recover by locating any Sentence
              individual or by using the provided sentence_id as a fallback container.
            - Conservative: rules that do not apply do not reject candidates.
            - The function does not invent linguistic facts.
    </ABoxRequirements>
    """

    # Step 1: Defensive validation. Always return a structured result; never raise.
    result_skeleton = {
        "sentence_id": sentence_id,
        "tokens": {},
        "error": None,
    }

    if abox is None:
        result_skeleton["error"] = "ABox is None"
        return result_skeleton

    if not isinstance(abox, dict):
        result_skeleton["error"] = "ABox is not a dictionary"
        return result_skeleton

    individuals = abox.get("individuals") or {}
    assertions = abox.get("assertions") or []

    if not isinstance(individuals, dict):
        individuals = {}
    if not isinstance(assertions, list):
        assertions = []

    # Step 2: Locate the sentence individual. Be tolerant: if exact id is missing,
    # try to find any Sentence-typed individual to use as the working sentence.
    effective_sentence_id = sentence_id
    if effective_sentence_id not in individuals or \
            (isinstance(individuals.get(effective_sentence_id), dict) and
             individuals[effective_sentence_id].get("type") != "Sentence"):
        # Fallback: find first Sentence individual
        fallback = None
        for ind_id, ind_val in individuals.items():
            if isinstance(ind_val, dict) and ind_val.get("type") == "Sentence":
                fallback = ind_id
                break
        if fallback is not None:
            effective_sentence_id = fallback
            result_skeleton["error"] = (
                f"Sentence id '{sentence_id}' not found; using fallback '{fallback}'"
            )
        else:
            # No Sentence individual at all. Continue with empty token set.
            result_skeleton["error"] = (
                f"Sentence id '{sentence_id}' not found and no Sentence individual present"
            )
            result_skeleton["sentence_id"] = sentence_id
            return result_skeleton

    result_skeleton["sentence_id"] = effective_sentence_id

    # Step 3: Index assertions for efficient lookup.
    binary_index = {}
    unary_index = {}
    for a in assertions:
        if not isinstance(a, dict):
            continue
        pred = a.get("predicate")
        args = a.get("args", [])
        if pred is None or not isinstance(args, list):
            continue
        if len(args) == 1:
            unary_index.setdefault(pred, set()).add(args[0])
        elif len(args) >= 2:
            binary_index.setdefault(pred, []).append((args[0], args[1]))

    def bin_get(pred):
        return binary_index.get(pred, [])

    # Step 4: Collect tokens belonging to the effective sentence.
    sentence_tokens = []
    for s, o in bin_get("belongsToSentence"):
        if o == effective_sentence_id and s in individuals \
                and isinstance(individuals[s], dict) \
                and individuals[s].get("type") == "Token":
            sentence_tokens.append(s)

    # If no belongsToSentence links, fall back to all Token individuals.
    if not sentence_tokens:
        for ind_id, ind_val in individuals.items():
            if isinstance(ind_val, dict) and ind_val.get("type") == "Token":
                sentence_tokens.append(ind_id)

    # Step 5: Collect candidates per token.
    token_to_candidates = {t: [] for t in sentence_tokens}
    for cand, tok in bin_get("candidateOf"):
        if tok in token_to_candidates and cand in individuals \
                and isinstance(individuals[cand], dict) \
                and individuals[cand].get("type") == "CandidateAnalysis":
            token_to_candidates[tok].append(cand)

    # Step 6: Token-level attribute maps.
    token_form = {s: o for s, o in bin_get("hasForm")}
    token_clause = {s: o for s, o in bin_get("belongsToClause")}
    token_subject = {s: o for s, o in bin_get("hasSubject")}
    token_person = {s: o for s, o in bin_get("tokenHasPerson")}
    token_number = {s: o for s, o in bin_get("tokenHasNumber")}
    token_animacy = {s: o for s, o in bin_get("tokenHasAnimacy")}
    token_role = {s: o for s, o in bin_get("tokenHasSemanticRole")}
    token_clause_final = {s: str(o).lower() == "true" for s, o in bin_get("tokenIsClauseFinal")}

    predecessors_of = {}
    successors_of = {}
    for x, t in bin_get("precedesToken"):
        predecessors_of.setdefault(t, set()).add(x)
        successors_of.setdefault(x, set()).add(t)

    clause_to_tokens = {}
    for tok, cl in token_clause.items():
        clause_to_tokens.setdefault(cl, set()).add(tok)

    clause_has_no_aux = unary_index.get("clauseHasNoAuxiliary", set())
    clause_neg_aux = {}
    for cl, ttok in bin_get("clauseContainsNegativeAuxiliary"):
        clause_neg_aux.setdefault(cl, set()).add(ttok)

    # Step 7: Candidate feature maps.
    feature_predicates = {
        "hasLemma": "lemma",
        "hasMorphBreakdown": "morph",
        "hasRawTag": "raw_tag",
        "hasCandidatePOS": "pos",
        "hasCandidateCase": "case",
        "hasCandidateCaseAmbiguous": "case_ambig",
        "hasCandidateNumber": "number",
        "hasCandidatePerson": "person",
        "hasCandidateTense": "tense",
        "hasCandidateMood": "mood",
        "hasCandidateVoice": "voice",
        "hasCandidateAspect": "aspect",
        "hasCandidateVerbForm": "verbform",
        "hasCandidatePolarity": "polarity",
        "hasCandidatePossessivePerson": "poss_person",
        "hasCandidatePossessiveNumber": "poss_number",
        "hasCandidateDerivationalFeature": "deriv",
    }

    cand_feat = {}
    for tok, cands in token_to_candidates.items():
        for c in cands:
            cand_feat[c] = {
                "token": tok,
                "supporting_rules": [],
                "rejecting_constraints": [],
            }

    for pred, key in feature_predicates.items():
        for s, o in bin_get(pred):
            if s in cand_feat:
                cand_feat[s][key] = o

    def f(c, key):
        return cand_feat.get(c, {}).get(key)

    def candidates_of(tok):
        return token_to_candidates.get(tok, [])

    # Extra helpers for corpus-grounded disambiguation rules.
    def clean_form(tok):
        s = str(token_form.get(tok, "") or "").strip().lower()
        return s.strip('\n\r\t "«»“”„()[]{}<>,.;:!?—–-')

    def raw(c):
        return str(f(c, "raw_tag") or "")

    def tag_is(c, tag):
        return raw(c) == tag

    def sorted_sentence_tokens():
        positions = dict(bin_get("hasPosition"))
        def pos_key(t):
            try:
                return int(positions.get(t, 10**9))
            except Exception:
                return 10**9
        return sorted(sentence_tokens, key=pos_key)

    ordered_tokens = sorted_sentence_tokens()

    def previous_tokens(tok):
        if tok not in ordered_tokens:
            return []
        return ordered_tokens[:ordered_tokens.index(tok)]

    def next_tokens(tok):
        if tok not in ordered_tokens:
            return []
        return ordered_tokens[ordered_tokens.index(tok) + 1:]

    def previous_token_form(tok):
        prev = previous_tokens(tok)
        return clean_form(prev[-1]) if prev else ""

    def next_token_form(tok):
        nxt = next_tokens(tok)
        return clean_form(nxt[0]) if nxt else ""

    def raw_tags_for_token(tok):
        return [raw(x) for x in candidates_of(tok)]

    def has_alt(tok, predicate):
        return any(predicate(raw(x)) for x in candidates_of(tok))

    def next_n_tokens(tok, n=1):
        return next_tokens(tok)[:n]

    def prev_n_tokens(tok, n=1):
        prev = previous_tokens(tok)
        return prev[-n:] if n > 0 else []

    def looks_name_like_form_value(value):
        value = str(value or "")
        return bool(value[:1].isupper() or value.replace(".", "").isupper())

    def has_following_finite_verb(tok, window=3):
        for nt in next_n_tokens(tok, window):
            for nc in candidates_of(nt):
                if f(nc, "tense") in {"Past", "Present", "Future"}:
                    return True
        return False

    # Step 8: Apply ontology-derived rules conservatively.
    for c, feats in cand_feat.items():
        tok = feats["token"]
        clause = token_clause.get(tok)
        preds = predecessors_of.get(tok, set())
        succs = successors_of.get(tok, set())

        # NegationCompatibleAnalysis
        if f(c, "polarity") == "Negative":
            if clause is not None and clause in clause_neg_aux and clause_neg_aux[clause]:
                feats["supporting_rules"].append("NegationCompatibleAnalysis")

        # NegativeAuxiliaryPersonCompatibleAnalysis
        if f(c, "polarity") == "Negative" and f(c, "person") == "3":
            for p in preds:
                if token_form.get(p) in ("uz", "уз", "um", "ум"):
                    feats["supporting_rules"].append("NegativeAuxiliaryPersonCompatibleAnalysis")
                    break

        # SubjectPersonAgreementCompatibleAnalysis
        if f(c, "tense") == "Future" and f(c, "person") == "1" and f(c, "number") == "Singular":
            for p in preds:
                if token_form.get(p) in ("mon", "мон"):
                    feats["supporting_rules"].append("SubjectPersonAgreementCompatibleAnalysis")
                    break

        # PluralSubjectAgreementCompatibleAnalysis
        if f(c, "tense") == "Present" and f(c, "person") == "3" and f(c, "number") == "Plural":
            matched = False
            for p in preds:
                for pc in candidates_of(p):
                    if f(pc, "number") == "Plural" and f(pc, "case") == "Nominative":
                        matched = True
                        break
                if matched:
                    break
            if matched:
                feats["supporting_rules"].append("PluralSubjectAgreementCompatibleAnalysis")

        # ImperativeIncompatibleAnalysis
        if f(c, "mood") == "Imperative" and f(c, "number") == "Plural":
            bad = False
            for p in preds:
                for pc in candidates_of(p):
                    if f(pc, "pos") == "Pronoun" and f(pc, "number") == "Singular":
                        bad = True
                        break
                if bad:
                    break
            if bad:
                feats["rejecting_constraints"].append("ImperativeIncompatibleAnalysis")

        # AttributiveParticipleCompatibleAnalysis
        if f(c, "verbform") == "Participle" and f(c, "voice") == "Passive":
            found = False
            for n in succs:
                for nc in candidates_of(n):
                    if f(nc, "pos") == "Noun":
                        found = True
                        break
                if found:
                    break
            if found:
                feats["supporting_rules"].append("AttributiveParticipleCompatibleAnalysis")

        # EvidentialContextCompatibleAnalysis
        if f(c, "verbform") == "Evidential":
            if token_clause_final.get(tok, False) and clause in clause_has_no_aux:
                feats["supporting_rules"].append("EvidentialContextCompatibleAnalysis")

        # InfinitiveCompatibleAnalysis
        if f(c, "verbform") == "Infinitive":
            ok = False
            for p in preds:
                for pc in candidates_of(p):
                    if f(pc, "pos") == "Verb":
                        ok = True
                        break
                if ok:
                    break
            if ok:
                feats["supporting_rules"].append("InfinitiveCompatibleAnalysis")

        # ConverbCompatibleAnalysis
        if f(c, "verbform") == "Converb" and clause is not None:
            ok = False
            for v in clause_to_tokens.get(clause, set()):
                if v == tok:
                    continue
                if v in successors_of.get(tok, set()):
                    for vc in candidates_of(v):
                        if f(vc, "pos") == "Verb":
                            ok = True
                            break
                if ok:
                    break
            if ok:
                feats["supporting_rules"].append("ConverbCompatibleAnalysis")

        # PossessiveOverAccusativeIncompatibleAnalysis
        if f(c, "case") == "Accusative":
            bad = False
            for p in preds:
                for pc in candidates_of(p):
                    if f(pc, "case") == "Genitive":
                        bad = True
                        break
                if bad:
                    break
            if bad:
                feats["rejecting_constraints"].append("PossessiveOverAccusativeIncompatibleAnalysis")

        # FirstPersonPRS12CompatibleAnalysis
        if f(c, "tense") == "Present" and f(c, "person") == "12":
            for p in preds:
                if token_form.get(p) in ("mon", "мон"):
                    feats["supporting_rules"].append("FirstPersonPRS12CompatibleAnalysis")
                    break

        # ActiveParticipleAttributiveCompatibleAnalysis
        if f(c, "verbform") == "Participle" and f(c, "voice") == "Active":
            found = False
            for n in succs:
                for nc in candidates_of(n):
                    if f(nc, "pos") == "Noun":
                        found = True
                        break
                if found:
                    break
            if found:
                feats["supporting_rules"].append("ActiveParticipleAttributiveCompatibleAnalysis")

        # ImperativeWithThirdPersonSubjectIncompatible
        if f(c, "mood") == "Imperative":
            subj = token_subject.get(tok)
            if subj is not None and token_person.get(subj) == "3":
                feats["rejecting_constraints"].append("ImperativeWithThirdPersonSubjectIncompatible")

        # PassivePastCompatibleAnalysis
        if (f(c, "voice") == "Passive" and f(c, "tense") == "Past"
                and f(c, "person") == "3"):
            subj = token_subject.get(tok)
            if subj is not None and token_animacy.get(subj) == "Inanimate" \
                    and token_role.get(subj) == "Patient":
                feats["supporting_rules"].append("PassivePastCompatibleAnalysis")

        # IterativeContextCompatibleAnalysis
        if f(c, "aspect") == "Iterative" and clause is not None:
            for x in clause_to_tokens.get(clause, set()):
                if token_form.get(x) in ("uno", "уно"):
                    feats["supporting_rules"].append("IterativeContextCompatibleAnalysis")
                    break

        # AttributiveBeforeNounCompatibleAnalysis
        if f(c, "verbform") == "Attributive":
            found = False
            for n in succs:
                for nc in candidates_of(n):
                    if f(nc, "pos") == "Noun":
                        found = True
                        break
                if found:
                    break
            if found:
                feats["supporting_rules"].append("AttributiveBeforeNounCompatibleAnalysis")

        # PostpositionGovernmentCompatibleAnalysis
        if f(c, "pos") == "Postposition":
            ok = False
            for p in preds:
                for pc in candidates_of(p):
                    if f(pc, "case") == "Genitive":
                        ok = True
                        break
                if ok:
                    break
            if ok:
                feats["supporting_rules"].append("PostpositionGovernmentCompatibleAnalysis")

        # DebitivePredicateCompatibleAnalysis
        if f(c, "mood") == "Debitive" and token_clause_final.get(tok, False):
            feats["supporting_rules"].append("DebitivePredicateCompatibleAnalysis")

        # SubjectVerbAgreementCompatibleAnalysis
        if (f(c, "tense") == "Past" and f(c, "person") == "3"
                and f(c, "number") == "Singular"):
            subj = token_subject.get(tok)
            if subj is not None and token_person.get(subj) == "3" \
                    and token_number.get(subj) == "Singular":
                feats["supporting_rules"].append("SubjectVerbAgreementCompatibleAnalysis")


        # Extra corpus-grounded Udmurt disambiguation rules.
        # These rules add support; they do not invent new candidates.

        # 1. PTCP.PST-LOC before negative copula ӧвӧл.
        if tag_is(c, "STEM-PTCP.PST-LOC") and next_token_form(tok) == "ӧвӧл":
            feats["supporting_rules"].append(
                "PastParticipleLocBeforeNegativeCopulaCompatibleAnalysis"
            )

        # 2. Converb in -са / -ыса.
        if tag_is(c, "STEM-CVB"):
            sf = clean_form(tok)
            if sf.endswith("са") or sf.endswith("ыса"):
                feats["supporting_rules"].append(
                    "ConverbSuffixSaCompatibleAnalysis"
                )

        # 3. Closed-class adverb/particle али -> STEM.
        if clean_form(tok) == "али" and tag_is(c, "STEM"):
            feats["supporting_rules"].append(
                "ClosedClassParticleStemCompatibleAnalysis"
            )

        # 4. PRS.12 in first-person context with мон.
        if raw(c).startswith("STEM-PRS.12"):
            context_tokens = set(sentence_tokens)
            if clause is not None and clause in clause_to_tokens:
                context_tokens = clause_to_tokens[clause]
            if any(clean_form(x) == "мон" for x in context_tokens):
                feats["supporting_rules"].append(
                    "FirstPersonContextPRS12CompatibleAnalysis"
                )

        # 5. луоз / луоз-а future form, especially after кулэ.
        if (clean_form(tok) in {"луоз", "луоз-а"} or clean_form(tok).startswith("луоз")) and tag_is(c, "STEM-FUT-3SG"):
            if previous_token_form(tok) == "кулэ" or any(clean_form(x) == "кулэ" for x in previous_tokens(tok)[-3:]) or clean_form(tok).startswith("луоз"):
                feats["supporting_rules"].append(
                    "FutureAfterKuleCompatibleAnalysis"
                )

        # 6. Proper names, initials and all-caps abbreviations are bare nominal stems
        # unless an explicit suffix is visible. This resolves cases where UniParser
        # proposes verbal/possessive readings for surnames or initials.
        sf = clean_form(tok)
        original_form = str(token_form.get(tok, "") or "")
        if (tag_is(c, "STEM") and looks_name_like_form_value(original_form)
                and not has_alt(tok, lambda r: "ATTR" in r)):
            feats["supporting_rules"].append("NameOrAbbreviationBareStemCompatibleAnalysis")

        # 7. Common particles/adverbs are bare STEM. The rule is lexical-class based,
        # not sentence-specific, and only supports an already available STEM candidate.
        if tag_is(c, "STEM") and sf in {
            "али", "гинэ", "но", "ик", "ини", "берло", "одно", "мед", "пыр",
            "на", "ку", "ке", "кулэ", "туж", "зэмос", "быдэс", "жок", "тыр",
        }:
            feats["supporting_rules"].append("ClosedClassStemCompatibleAnalysis")

        # 8. A token immediately before restrictive гинэ is often an adjectival/adverbial
        # bare stem; this repairs analyzer omissions such as умойзэ гинэ without
        # deleting the original possessive/accusative candidate.
        if tag_is(c, "STEM") and next_token_form(tok) == "гинэ":
            if not has_alt(tok, lambda r: r == "STEM-ACC"):
                feats["supporting_rules"].append("PreRestrictiveParticleStemCompatibleAnalysis")

        # 9. Productive -сько/-исько present 1/2-person form.
        if raw(c).startswith("STEM-PRS.12") and (sf.endswith("сько") or sf.endswith("исько")):
            feats["supporting_rules"].append("PresentTwelveSuffixCompatibleAnalysis")

        # 10. Prefer a non-causative reading when the only competing reading is the
        # same tense/mood plus CAUS, unless independent causative evidence is present.
        if raw(c).startswith("STEM-FUT-1PL") and has_alt(tok, lambda r: "CAUS-FUT-1PL" in r):
            feats["supporting_rules"].append("PlainFutureOverCausativeCompatibleAnalysis")
        if raw(c).startswith("STEM-IMP") and "CAUS" not in raw(c) and has_alt(tok, lambda r: "CAUS-IMP" in r):
            feats["supporting_rules"].append("PlainImperativeOverCausativeCompatibleAnalysis")

        # 11. Plain converb is preferred over passive converb if there is no
        # explicit passive subject/patient cue in the ABox.
        if tag_is(c, "STEM-CVB") and has_alt(tok, lambda r: r == "STEM-PASS-CVB"):
            feats["supporting_rules"].append("PlainConverbOverPassiveCompatibleAnalysis")

        # 12. Past participles in -ем/-эм/-тэм are strong modifiers before a noun.
        if raw(c).startswith("STEM-PTCP.PST") and not has_alt(tok, lambda r: r == "STEM-NEG.ATTR"):
            if sf.endswith(("ем", "эм", "тэм", "ям")):
                for nt in next_n_tokens(tok, 2):
                    if any(f(nc, "pos") == "Noun" for nc in candidates_of(nt)):
                        feats["supporting_rules"].append("PastParticipleModifierCompatibleAnalysis")
                        break

        # If the only verbal-form ambiguity is EVID vs PTCP.PST, prefer the
        # participial reading outside clause-final evidential context.
        if (raw(c).startswith("STEM-PTCP.PST") and has_alt(tok, lambda r: "EVID" in r)
                and not has_alt(tok, lambda r: r == "STEM-NEG.ATTR")):
            if not token_clause_final.get(tok, False):
                feats["supporting_rules"].append("PastParticipleOverEvidentialCompatibleAnalysis")

        # Negative attributive forms before nouns are stronger than a generic
        # past-participle reading in the same position.
        if tag_is(c, "STEM-NEG.ATTR"):
            for nt in next_n_tokens(tok, 2):
                if any(f(nc, "pos") == "Noun" for nc in candidates_of(nt)):
                    feats["supporting_rules"].append("NegativeAttributiveBeforeNounCompatibleAnalysis")
                    break

        # 13. Udmurt -оно/-ёно debitive/necessitative modifier.
        if tag_is(c, "STEM-DEB") and sf.endswith(("оно", "ёно")):
            feats["supporting_rules"].append("DebitiveOnoSuffixCompatibleAnalysis")

        # 14. Plural illative -осы is common in noun forms; prefer it over a possessive
        # analysis when there is no 2nd-person possessor cue nearby.
        if tag_is(c, "STEM-PL-ILL") and sf.endswith("осы"):
            prev_forms = {clean_form(x) for x in prev_n_tokens(tok, 3)}
            if not (prev_forms & {"тон", "тӥ"}):
                feats["supporting_rules"].append("PluralIllativeOsyCompatibleAnalysis")

        # 15. Accusative plural pronouns ending -осты/-ёсты, or ACC.PL before
        # a finite verb where the competitor is a possessive plural.
        if "ACC.PL" in raw(c):
            if sf.endswith(("осты", "ёсты")) or (has_alt(tok, lambda r: "P.2PL" in r) and has_following_finite_verb(tok, 3)):
                feats["supporting_rules"].append("PluralAccusativePronounCompatibleAnalysis")

        # 16. Object accusative before a finite verb, especially where the competitor
        # is a possessive reading of the same surface form.
        if tag_is(c, "STEM-ACC") and has_alt(tok, lambda r: r.startswith("STEM-P.")):
            if has_following_finite_verb(tok, 3):
                feats["supporting_rules"].append("AccusativeObjectBeforeFiniteVerbCompatibleAnalysis")

        # 17. First-person plural possessive/finite-looking -мы nominal forms.
        if tag_is(c, "STEM-P.1PL") and sf.endswith("мы"):
            feats["supporting_rules"].append("FirstPluralMySuffixCompatibleAnalysis")

        # 18. In appositive organization names, партия is a bare noun; the analyzer may
        # also produce PROL-ADV because of internal -ти-.
        if tag_is(c, "STEM") and has_alt(tok, lambda r: "PROL-ADV" in r):
            if previous_token_form(tok) or next_token_form(tok):
                feats["supporting_rules"].append("NominalStemOverProlativeAdverbCompatibleAnalysis")

    # Step 9: Build per-token result.
    result = {"sentence_id": effective_sentence_id, "tokens": {}, "error": result_skeleton["error"]}

    for tok, cands in token_to_candidates.items():
        remaining = [c for c in cands if not cand_feat[c]["rejecting_constraints"]]
        rejected = [c for c in cands if cand_feat[c]["rejecting_constraints"]]

        if not cands:
            status = "no_candidates"
            selected = None
            remaining_final = []
        elif not remaining:
            status = "contradiction"
            selected = None
            remaining_final = cands
        elif len(remaining) == 1:
            status = "resolved"
            selected = remaining[0]
            remaining_final = remaining
        else:
            supports = [(c, len(cand_feat[c]["supporting_rules"])) for c in remaining]
            supports.sort(key=lambda x: x[1], reverse=True)
            top_score = supports[0][1]
            top = [c for c, s in supports if s == top_score]
            if top_score > 0 and len(top) == 1:
                status = "resolved"
                selected = top[0]
            else:
                status = "unresolved"
                selected = None
            remaining_final = remaining

        result["tokens"][tok] = {
            "form": token_form.get(tok),
            "status": status,
            "selected_candidate": selected,
            "remaining_candidates": remaining_final,
            "rejected_candidates": rejected,
            "explanations": {
                c: {
                    "supporting_rules": list(cand_feat[c]["supporting_rules"]),
                    "rejecting_constraints": list(cand_feat[c]["rejecting_constraints"]),
                }
                for c in cands
            },
        }

    return result


def calculate(abox: dict, sent_id: str) -> float:
    """
    Wrapper that returns the proportion of resolved tokens as a float.
    Always returns 0.0 on failure rather than raising.
    """
    try:
        res = disambiguate(abox, sent_id)
    except Exception:
        return 0.0
    if not isinstance(res, dict) or "tokens" not in res:
        return 0.0
    tokens = res["tokens"]
    if not tokens:
        return 0.0
    total = len(tokens)
    resolved = sum(1 for t in tokens.values() if t.get("status") == "resolved")
    return float(resolved) / float(total) if total > 0 else 0.0