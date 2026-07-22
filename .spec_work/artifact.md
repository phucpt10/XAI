# Template execution contract

## Reference

- Retained DOCX: `D:\ResearchCode\XAI\Template_DacTa_ResearchSoftware.docx`
- SHA-256: `44941816fee5f0c6e4d76a0157b3d11fd1bd0e4b767f6ff779fbefa031944b61`
- Sections: 1
- Paragraphs: 167; tables: 11; inline shapes: 0
- Page count: unresolved by rendering because LibreOffice/soffice is unavailable. The source contains explicit page breaks corresponding to an expected 14-page pattern; this is not treated as a verified rendered page count.
- Structural evidence: `.spec_work/template-style-evidence.json`, section/style/heading audit output, and `.spec_work/inspect_template.py` output.
- Render evidence: attempted at `.spec_work/reference_render`; failed before conversion because `soffice` was not installed.

## Page system

- One portrait US Letter section, 8.50 x 11.00 inches.
- Margins: left 1.25 in, right 1.25 in, top 1.00 in, bottom 1.00 in.
- Usable body width: 6.00 in (8640 DXA).
- No distinct first/odd/even-page headers. Header and footer each contain only an empty paragraph.
- Major A-J sections use explicit page breaks; subsections flow naturally.

## Typography and components

- Theme body type is the source Word theme (Calibri-compatible minor font); body text remains 11 pt unless a source component specifies otherwise.
- Title: source `Title`/Heading 0 role, 26 pt, dark navy `17365D`, centered, single spacing, 15 pt after.
- Subtitle: 12 pt, italic, blue `4F81BD`, centered.
- Heading 1: 14 pt, bold, blue family; the source applies direct dark-blue `1F4E79` to visible heading runs, 24 pt before, keep-with-next.
- Heading 2: 13 pt, bold, blue family; source visible heading runs use `1F4E79`, 10 pt before, keep-with-next.
- Heading 3: bold, blue family, 10 pt before, keep-with-next.
- Code/pseudocode block: Consolas 9 pt, 0.30 in left indent, paragraph fill `F2F2F2`.
- Tables: source `Light Grid Accent 1` visual system; dark-blue header fill `1F4E79`, bold white header text, alternating/light grid body. New tables must use explicit 6.00 in geometry, content-weighted columns, 0.08 in cell padding, vertically centered cells, repeating header rows, and no fixed row heights.
- Lists: real Word `List Bullet` and `List Number` styles; no Unicode/manual markers.
- Page furniture: none beyond title block and body components.

## Content flow

1. Cover page: title, subtitle, document status, version, date, audience, protocol state.
2. Design philosophy and status legend.
3. A - Research Context.
4. B - Scope and Boundaries.
5. C - System Architecture.
6. D - Module Specifications.
7. E - Data Contracts.
8. F - Business Rules.
9. G - Scientific Invariants.
10. H - Quality Gates.
11. I - Testing Strategy.
12. J - Handoff Protocol.
13. Traceability matrix, decision log, unresolved items, and conclusion.

## Slot map

- `word/document.xml` body: fully editable and must be rewritten in English for PlantXAI-Stability. The source body is a template/example, not content to preserve.
- Section properties at the end of `word/document.xml`: preserve page size, margins, orientation, header/footer distances, and single-section behavior.
- `word/styles.xml`, `word/stylesWithEffects.xml`, `word/theme/theme1.xml`, `word/fontTable.xml`, `word/numbering.xml`: preserve as design authority; new content reuses these styles/definitions.
- `word/settings.xml`, `word/webSettings.xml`: preserve except a settings-only update-field flag is permitted if fields are added. No TOC or fields are required for this deliverable.
- `customXml/*`: preserve-only.
- `word/_rels/document.xml.rels` and package relationships: preserve unless a required new relationship is added. This specification requires no images or external links.
- `docProps/core.xml` and `docProps/app.xml`: metadata may change as a normal result of authoring; no personal/sensitive data should be added.
- `docProps/thumbnail.jpeg`: preserve-only where possible.

## Package preservation

- Preserve all source parts except `word/document.xml` and ordinary document metadata.
- The output must be created from a copy of the reference, not from a blank file.
- The source template must still match the recorded SHA-256 after authoring.

## Fidelity gates

- Same single-section Letter geometry and margins.
- Same title/heading blue hierarchy, table header treatment, and grey code-block component.
- Real heading and list styles; no fake headings or bullets.
- Explicit table widths and cell widths; repeated header rows; no clipped fixed-height rows.
- All placeholders removed. Provisional values are labeled `PROVISIONAL`; unresolved scientific choices are labeled `PENDING` or `BLOCKED`.
- No fabricated dataset counts, model results, p-values, confidence intervals, or experimental conclusions.
- Final structural audits must check sections, headings, styles, tables, placeholders, protocol state, and reference hash.
- Preferred PNG render/diff gate must be attempted. If it remains unavailable solely because `soffice` is missing, complete structural QA and disclose that visual QA was not available.
