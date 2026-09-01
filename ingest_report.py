"""
Ingest a new month's Flash Report PDF into the growing master dataset.
=========================================================================
Usage:  python ingest_report.py <path_to_pdf> <YYYY-MM>
Example: python ingest_report.py FlashReport_July_2026.pdf 2026-07

Appends parsed rows into data/projects_master.csv, tagged with report_month,
and de-dupes on (project_name, state, original_cost_cr) so re-running on
the same file twice doesn't create duplicates.
"""

import sys
import re
import subprocess
import pandas as pd

AGENCY_RE = re.compile(r'\)\s+(\d{2}/\d{4})\s+(\d{2}/\d{4})\s+([\d,]+\.?\d*)\s*$')
REVISED_RE = re.compile(r'\((\d{2}/\d{4})\)\s+\((\d{2}/\d{4})\)\s+\(([\d,]+\.?\d*)\)\s*$')
REVISED_RE_DASH = re.compile(r'\((\d{2}/\d{4})\)\s+\(-\)\s+\(([\d,]+\.?\d*)\)\s*$')
SLNO_STATE_RE = re.compile(r'^\s*(\d+)?\s+([A-Za-z][A-Za-z .&]+?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)\s*$')
MINISTRY_RE = re.compile(r'^\s*Ministry of ')
PAGEJUNK_RE = re.compile(
    r'Project Assessment, Infrastructure|^\x0c|^\s*Page\s|For details visit|'
    r'^\(PAIMANA\)|^\s*Sl\.No|^\s*All Ongoing Projects'
)


def clean_num(s):
    if s is None:
        return None
    return float(s.replace(',', ''))


def extract_text(pdf_path):
    txt_path = "/tmp/_ingest_flash.txt"
    subprocess.run(["pdftotext", "-layout", pdf_path, txt_path], check=True)
    with open(txt_path) as f:
        return f.readlines()


def find_table_bounds(lines):
    start = end = None
    for i, l in enumerate(lines):
        if "All Ongoing Projects" in l and start is None:
            start = i
        if start is not None and re.search(r'Total\s*\(\d+\)', l):
            end = i
            break
    if start is None:
        raise ValueError("Could not find 'All Ongoing Projects' table in this PDF.")
    return start, end or len(lines)


def parse_ongoing_table(lines, start_idx, end_idx):
    records = []
    current_ministry = None
    current_sector = None
    name_buffer = []
    i = start_idx

    while i < min(end_idx, len(lines)):
        line = lines[i].rstrip('\n')

        if (PAGEJUNK_RE.search(line) or 'Table 6' in line or 'Project Name' in line
                or 'Date of Approval' in line or 'Orignal' in line or 'Cumulative' in line
                or 'MM/YYYY' in line or 'Physical Progress' in line
                or '(Project Code)' == line.strip()):
            i += 1
            continue

        if MINISTRY_RE.match(line):
            current_ministry = line.strip()
            current_sector = None
            name_buffer = []
            i += 1
            continue

        stripped = line.strip()
        if stripped == '':
            i += 1
            continue

        m_agency = AGENCY_RE.search(line)
        if m_agency:
            approval, target_doc, orig_cost = m_agency.groups()
            pre = line[:m_agency.start() + 1]
            project_name = ' '.join(x.strip() for x in name_buffer).strip()
            agency = pre.strip()

            slno = state = cum_exp = progress = None
            rev_start = rev_doc = rev_cost = None
            for j in range(i + 1, min(i + 5, len(lines))):
                l2 = lines[j].rstrip('\n')
                if l2.strip() == '' or MINISTRY_RE.match(l2):
                    break
                m_ss = SLNO_STATE_RE.match(l2)
                if m_ss and cum_exp is None:
                    slno, state, cum_exp, progress = m_ss.groups()
                    continue
                m_rev = REVISED_RE.search(l2)
                if m_rev:
                    rev_start, rev_doc, rev_cost = m_rev.groups()
                    continue
                m_revd = REVISED_RE_DASH.search(l2)
                if m_revd:
                    rev_start, rev_cost = m_revd.groups()
                    rev_doc = None

            records.append({
                'ministry': current_ministry, 'sector': current_sector,
                'project_name': project_name, 'agency': agency,
                'sl_no': slno, 'state': state,
                'approval_date': approval, 'start_date': rev_start,
                'target_doc': target_doc, 'revised_doc': rev_doc,
                'original_cost_cr': clean_num(orig_cost),
                'revised_cost_cr': clean_num(rev_cost) if rev_cost else clean_num(orig_cost),
                'cumulative_expenditure_cr': clean_num(cum_exp),
                'physical_progress_pct': clean_num(progress),
            })
            name_buffer = []
            i += 1
            continue

        if current_sector is None and current_ministry and not re.search(r'\d', line):
            current_sector = stripped
            i += 1
            continue

        if (re.match(r'^\s*\(-\)', line) or SLNO_STATE_RE.match(line)
                or REVISED_RE.search(line) or REVISED_RE_DASH.search(line)):
            i += 1
            continue

        name_buffer.append(stripped)
        i += 1

    return records


def main():
    if len(sys.argv) != 3:
        print("Usage: python ingest_report.py <pdf_path> <YYYY-MM>")
        sys.exit(1)

    pdf_path, report_month = sys.argv[1], sys.argv[2]

    print(f"Extracting text from {pdf_path} ...")
    lines = extract_text(pdf_path)
    start_idx, end_idx = find_table_bounds(lines)
    records = parse_ongoing_table(lines, start_idx, end_idx)
    print(f"Parsed {len(records)} raw rows.")

    df_new = pd.DataFrame(records)
    df_new = df_new.dropna(subset=['sl_no', 'state', 'cumulative_expenditure_cr', 'physical_progress_pct']).copy()
    df_new['cost_overrun_pct'] = (
        (df_new['revised_cost_cr'] - df_new['original_cost_cr']) / df_new['original_cost_cr'] * 100
    )
    df_new['report_month'] = report_month
    print(f"Clean rows this month: {len(df_new)}")

    master_path = "data/projects_master.csv"
    try:
        df_master = pd.read_csv(master_path)
    except FileNotFoundError:
        df_master = pd.DataFrame(columns=df_new.columns)

    combined = pd.concat([df_master, df_new], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=['project_name', 'state', 'original_cost_cr'], keep='last')
    print(f"Dropped {before - len(combined)} duplicate rows (already in master).")

    combined.to_csv(master_path, index=False)
    print(f"Master dataset now has {len(combined)} total project-month rows "
          f"across {combined['report_month'].nunique()} months.")


if __name__ == "__main__":
    main()
