import os
import re
import json
import logging
import argparse
import pandas as pd
import numpy as np


class EwEConverter:
    """
    A class to handle conversions between EwE (Ecopath with Ecosim)
    CSV/Excel input formats, JSON, and reconstructed Excel files.
    """

    def __init__(self, log_file="ewe_conversion.log"):
        self.logger = self._setup_logger(log_file)

    def _setup_logger(self, log_file):
        """Sets up a logger that outputs to both a file and the console."""
        logger = logging.getLogger("EwE_Converter")
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter('%(levelname)s - %(message)s')

        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    @staticmethod
    def _find_file(directory, keyword):
        """Finds a file in the given directory that contains the keyword in its name."""
        for f in os.listdir(directory):
            if keyword.lower() in f.lower() and not f.startswith('~$'):
                return os.path.join(directory, f)
        return None

    @staticmethod
    def _load_data(filepath, header='infer'):
        """Loads a CSV or Excel file into a pandas DataFrame."""
        if not filepath:
            return None
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath, header=header)
        elif filepath.endswith(('.xls', '.xlsx')):
            xl_header = 0 if header == 'infer' else header
            return pd.read_excel(filepath, header=xl_header)
        return None

    @staticmethod
    def _load_metadata(filepath):
        """Loads the metadata file and returns a dictionary of key-value pairs."""
        if not filepath: return {}
        try:
            df = EwEConverter._load_data(filepath, header=None)
            meta = {}
            for _, row in df.iterrows():
                if len(row) >= 2 and pd.notna(row.iloc[0]):
                    key = str(row.iloc[0]).strip().lower()
                    val = str(row.iloc[1]).strip()
                    meta[key] = val
            return meta
        except Exception:
            return {}

    @staticmethod
    def _fmt_float(f):
        """Safely formats floats to strings, removing trailing .0 if it's an integer."""
        try:
            if float(f) == int(float(f)):
                return str(int(float(f)))
            return str(float(f))
        except (ValueError, TypeError):
            return str(f)

    @staticmethod
    def _parse_metadata_from_filename(filename):
        """Attempts to extract Metadata from the LME_Number_Name_(Year).json format."""
        base = os.path.basename(filename).replace('.json', '')
        match = re.match(r"([^_]+)_(\d+)_([^_]+(?:_[^_]+)*)_\(([^)]+)\)", base)
        if match:
            return {
                'LME': match.group(1),
                'model_number': match.group(2),
                'model_name': match.group(3).replace('_', ' '),
                'model_year': match.group(4)
            }
        return {'Filename': base}

    def csv_to_json(self, input_dir):
        """
        Reads CSV/Excel EwE input files from a directory and converts them into a JSON model.
        Returns the path to the generated JSON file.
        """
        # 1. Locate Files
        files = {
            'basic': self._find_file(input_dir, 'basic input'),
            'diet': self._find_file(input_dir, 'diet composition'),
            'detritus': self._find_file(input_dir, 'detritus fate'),
            'discards': self._find_file(input_dir, 'discards'),
            'landings': self._find_file(input_dir, 'landings'),
            'tl': self._find_file(input_dir, 'tl'),
            'taxonomy': self._find_file(input_dir, 'taxonomy'),
            'metadata': self._find_file(input_dir, 'metadata')
        }

        # Update logger file path to current directory
        log_file = os.path.join(input_dir, "ewe_conversion.log")
        self.logger = self._setup_logger(log_file)
        self.logger.info(f"Starting conversion for directory: {input_dir}")

        # Check Mandatory Files
        if not files['basic']:
            self.logger.error("Mandatory file 'Basic input' not found. Aborting.")
            return None
        if not files['diet']:
            self.logger.error("Mandatory file 'Diet composition' not found. Aborting.")
            return None
        if not files['metadata']:
            self.logger.error("Mandatory file 'metadata' not found. Aborting.")
            return None

        # Handle Output Filename via Metadata
        meta_dict = self._load_metadata(files['metadata'])
        if meta_dict:
            lme = meta_dict.get('lme', 'LME')
            mod_num = meta_dict.get('model_number', '00000')
            mod_name = meta_dict.get('model_name', 'Model').replace(' ', '_')
            mod_year = meta_dict.get('model_year', 'Year')
            out_filename = f"{lme}_{mod_num}_{mod_name}_({mod_year}).json"
        else:
            out_filename = "ewe_model_output.json"

        output_json = os.path.join(input_dir, out_filename)
        self.logger.info(f"Output will be saved as: {out_filename}")

        # 2. Load DataFrames
        dfs = {k: self._load_data(v) for k, v in files.items() if k != 'metadata'}

        df_basic = dfs['basic'].dropna(how='all')
        df_diet = dfs['diet'].fillna(0) if dfs['diet'] is not None else None
        df_detritus = dfs['detritus']

        df_landings = dfs['landings'].set_index(dfs['landings'].columns[1]) if dfs[
                                                                                   'landings'] is not None else pd.DataFrame()
        df_discards = dfs['discards'].set_index(dfs['discards'].columns[1]) if dfs[
                                                                                   'discards'] is not None else pd.DataFrame()
        df_tax = dfs['taxonomy'].set_index(dfs['taxonomy'].columns[1]) if dfs[
                                                                              'taxonomy'] is not None else pd.DataFrame()

        # Pre-build a Name-to-Seq mapping
        name_to_seq = {}
        for idx, row in df_basic.iterrows():
            g_name = str(row.iloc[1]).strip()
            if pd.isna(row.iloc[0]) or g_name == 'nan':
                continue

            # IDENTIFICATION FIX: Skip Fleets/Fisheries
            # If Biomass (3), P/B (5), Q/B (6), and EE (7) are ALL NaN, it is not a biological group.
            if pd.isna(row.iloc[3]) and pd.isna(row.iloc[5]) and pd.isna(row.iloc[6]) and pd.isna(row.iloc[7]):
                continue

            raw_gseq = str(row.iloc[0]).strip()
            try:
                g_seq = str(int(float(raw_gseq)))
            except ValueError:
                g_seq = raw_gseq
            name_to_seq[g_name.lower()] = g_seq

        groups_json = []

        # 3. Process Each Group
        for idx, row in df_basic.iterrows():
            raw_gseq = str(row.iloc[0]).strip()
            try:
                group_seq = str(int(float(raw_gseq)))
            except ValueError:
                group_seq = raw_gseq

            group_name = str(row.iloc[1]).strip()

            if pd.isna(row.iloc[0]) or group_name == 'nan':
                continue

            # IDENTIFICATION FIX: Skip Fleets/Fisheries
            if pd.isna(row.iloc[3]) and pd.isna(row.iloc[5]) and pd.isna(row.iloc[6]) and pd.isna(row.iloc[7]):
                continue

            self.logger.info(f"Processing group: {group_name} (ID: {group_seq})")

            def get_val(val):
                if pd.isna(val) or str(val).strip() == '':
                    return "-9999", "false"
                return str(val), "true"

            biomass_hab_area, b_input = get_val(row.iloc[3])
            pb, pb_input = get_val(row.iloc[5])
            qb, qb_input = get_val(row.iloc[6])
            ee, ee_input = get_val(row.iloc[7])
            other_mort, _ = get_val(row.iloc[8] if len(row) > 8 else np.nan)
            gs, _ = get_val(row.iloc[10] if len(row) > 10 else np.nan)
            det_import, _ = get_val(row.iloc[11] if len(row) > 11 else np.nan)
            hab_area = str(row.iloc[2]) if not pd.isna(row.iloc[2]) else "1"
            biomass = str(float(biomass_hab_area) * float(hab_area))

            # A. Compile Real Diet FIRST
            raw_diet = []
            diet_sum = 0.0
            import_val = 0.0

            if df_diet is not None:
                df_diet_cols_str = [str(c).strip() for c in df_diet.columns]
                predator_col = None

                if group_seq in df_diet_cols_str:
                    predator_col = df_diet.columns[df_diet_cols_str.index(group_seq)]
                elif group_name in df_diet_cols_str:
                    predator_col = df_diet.columns[df_diet_cols_str.index(group_name)]

                if predator_col is not None:
                    for diet_idx, diet_row in df_diet.iterrows():
                        raw_prey_seq = str(diet_row.iloc[0]).strip()
                        prey_name = str(diet_row.iloc[1]).strip().lower() if len(diet_row) > 1 else ""

                        try:
                            proportion = float(diet_row[predator_col])
                        except (ValueError, TypeError, KeyError):
                            proportion = 0.0

                        try:
                            prey_seq = str(int(float(raw_prey_seq)))
                        except ValueError:
                            prey_seq = raw_prey_seq.lower()

                        if prey_seq == 'import' or prey_name == 'import':
                            import_val = proportion
                            continue

                        if prey_seq in ['sum', '1-sum', '0'] or prey_name in ['sum', '1-sum']:
                            continue

                        if proportion > 0:
                            raw_diet.append({
                                "prey_seq": prey_seq,
                                "proportion_val": proportion
                            })
                            diet_sum += proportion

            # B. Diet & Import Normalization
            total_diet_sum = diet_sum + import_val
            norm_factor = 1.0

            if total_diet_sum > 0:
                self.logger.info(f"    - Total Diet + Import sum for {group_name}: {total_diet_sum:.5f}")
                needs_normalization = not np.isclose(total_diet_sum, 1.0, atol=1e-6)
                if needs_normalization:
                    self.logger.warning(f"    - Diet Sum is not 1 ({total_diet_sum:.5f}). Normalizing to 1.0")
                norm_factor = 1.0 / total_diet_sum if needs_normalization else 1.0
                import_val = import_val * norm_factor

            # C. Merge Diet and Detritus Fate
            group_diet_map = {}

            # 1. Insert normalized real prey items
            for d in raw_diet:
                final_prop = d["proportion_val"] * norm_factor
                group_diet_map[d["prey_seq"]] = {
                    "proportion": final_prop,
                    "detritus_fate": 0.0
                }

            # 2. Insert Detritus Fate (Routing Proportions)
            if df_detritus is not None:
                det_row = None
                match_by_seq = df_detritus[
                    df_detritus.iloc[:, 0].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == group_seq]
                if not match_by_seq.empty:
                    det_row = match_by_seq.iloc[0]
                else:
                    match_by_name = df_detritus[
                        df_detritus.iloc[:, 1].astype(str).str.strip().str.lower() == group_name.lower()]
                    if not match_by_name.empty:
                        det_row = match_by_name.iloc[0]

                if det_row is not None:
                    for col in df_detritus.columns:
                        col_str = str(col).strip().lower()
                        if col_str in name_to_seq:
                            try:
                                fate_val = float(det_row[col])
                            except:
                                fate_val = 0.0

                            if fate_val > 0:
                                p_seq = name_to_seq[col_str]
                                if p_seq not in group_diet_map:
                                    group_diet_map[p_seq] = {"proportion": 0.0, "detritus_fate": fate_val}
                                else:
                                    group_diet_map[p_seq]["detritus_fate"] = fate_val

            # 3. Validate and Normalize Detritus Fate sum
            if df_detritus is not None and len(group_diet_map) > 0:
                det_fate_sum = sum(vals["detritus_fate"] for vals in group_diet_map.values())

                self.logger.info(f"    - Detritus fate sum for {group_name}: {det_fate_sum:.5f}")
                if det_fate_sum > 0 and not np.isclose(det_fate_sum, 1.0, atol=1e-6):
                    self.logger.warning(f"    - Detritus fate sum is not 1 ({det_fate_sum:.5f}). Normalizing to 1.0")
                    det_norm_factor = 1.0 / det_fate_sum
                    for p_seq in group_diet_map:
                        group_diet_map[p_seq]["detritus_fate"] *= det_norm_factor
                elif det_fate_sum == 0.0:
                    self.logger.warning(f"    - Missing Detritus fate routing for {group_name} (Sum = 0.0)")

            # 4. Convert Map to JSON list
            diet_list = []
            for p_seq, vals in group_diet_map.items():
                diet_list.append({
                    "prey_seq": p_seq,
                    "proportion": self._fmt_float(vals["proportion"]),
                    "detritus_fate": self._fmt_float(vals["detritus_fate"])
                })

            # D. Primary Producer & Detritus Identification
            is_detritus_group = False
            if df_detritus is not None:
                det_cols = [str(c).strip().lower() for c in df_detritus.columns]
                if group_name.lower() in det_cols:
                    is_detritus_group = True
            elif any(keyword in group_name.lower() for keyword in ["detritus", "offal", "discard", "carcass"]):
                is_detritus_group = True

            is_pp_candidate = (qb_input == "false")
            has_diet = (total_diet_sum > 0)

            if is_detritus_group:
                pp_flag = "2"
                self.logger.info(f"    -> Identified as Detritus group (pp = 2).")
            elif is_pp_candidate and not has_diet:
                pp_flag = "1"
                self.logger.info(f"    -> Identified as Primary Producer (pp = 1).")
            elif is_pp_candidate and has_diet:
                pp_flag = "0"
                self.logger.warning(f"    -> VALIDATION CONFLICT: Missing Q/B value, but has diet inputs!")
            elif not is_pp_candidate and not has_diet:
                pp_flag = "0"
                self.logger.warning(f"    -> VALIDATION CONFLICT: Has Q/B value ({qb}), but diet is empty!")
            else:
                pp_flag = "0"

            # E. Log missing parameters
            if b_input == "false": self.logger.warning(f"    - Missing Biomass for {group_name}")
            if pb_input == "false" and pp_flag != "2": self.logger.warning(f"    - Missing P/B for {group_name}")
            if ee_input == "false": self.logger.warning(f"    - Missing EE for {group_name}")
            if qb_input == "false" and pp_flag == "0": self.logger.warning(f"    - Missing Q/B for {group_name}")

            # Calculate Export (Landings + Discards)
            export_val = 0.0
            if group_name in df_landings.index and 'Total' in df_landings.columns:
                export_val += float(df_landings.loc[group_name, 'Total'] or 0)
            if group_name in df_discards.index and 'Total' in df_discards.columns:
                export_val += float(df_discards.loc[group_name, 'Total'] or 0)

            # Taxonomy Description
            taxon_descr = None
            if not df_tax.empty and group_name in df_tax.index:
                taxon_descr = str(df_tax.loc[group_name].iloc[-1])

            # Assemble Group JSON Object
            group_obj = {
                "group_name": group_name,
                "group_seq": group_seq,
                "habitat_area": hab_area,
                "biomass_habitat_area": biomass_hab_area,
                "b_hab_area_input": b_input,
                "biomass": biomass,
                "vbk": "0",
                "pb": pb,
                "pb_input": pb_input,
                "ee": ee,
                "ee_input": ee_input,
                "biomass_accum": "-9999",
                "biomass_accum_rate": "-9999",
                "qb": qb,
                "qb_input": qb_input,
                "pp": pp_flag,
                "detritus_import": det_import if det_import != "-9999" else "0",
                "respiration": "-9999",
                "immigration": "-9999",
                "emigration": "-9999",
                "emigration_rate": "-9999",
                "other_mort": other_mort,
                "export": self._fmt_float(export_val) if export_val > 0 else "0",
                "gs": gs if gs != "-9999" else "0",
                "shadow_price": "0",
                "ge": "-9999",
                "ge_input": "false",
                "diet_imp": self._fmt_float(import_val) if import_val > 0 else "0",
                "diet_descr": {"diet": diet_list if len(diet_list) > 1 else (diet_list[0] if diet_list else None)},
                "taxon_descr": taxon_descr,
                "pedigree_assignment_descr": None
            }

            if not group_obj["diet_descr"]["diet"]:
                group_obj["diet_descr"] = None

            groups_json.append(group_obj)

        # 4. Save to JSON
        final_json = {"group": groups_json}
        with open(output_json, 'w') as f:
            json.dump(final_json, f, indent=4)

        self.logger.info(f"\nSuccess: Output saved to {output_json}")
        self.logger.info(f"Log saved to {log_file}")
        return output_json

    def json_to_excel(self, json_path, output_excel=None):
        """
        Reads a JSON EwE model and reconstructs it into a multi-sheet Excel file.
        """
        if output_excel is None:
            output_excel = json_path.replace('.json', '_reconstructed.xlsx')

        self.logger.info(f"Reading JSON: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        groups = data.get('group', [])
        if not groups:
            self.logger.error("Error: No groups found in JSON.")
            return None

        # --- Prepare Trackers & Data Structures ---
        basic_rows = []
        tax_rows = []
        tl_rows = []
        landings_rows = []

        group_ids = [g['group_seq'] for g in groups]
        group_names = {g['group_seq']: g['group_name'] for g in groups}

        # Matrices
        diet_matrix = pd.DataFrame(0.0, index=group_ids + ['Import'], columns=group_ids)
        det_fate_matrix = pd.DataFrame(0.0, index=group_ids, columns=group_ids)

        has_tax = False
        has_tl = False
        has_export = False
        detritus_pools = set()

        def parse_val(val):
            """Converts -9999 back to a blank/None representation"""
            if pd.isna(val) or val == "-9999" or val is None:
                return ""
            try:
                return float(val) if '.' in str(val) else int(val)
            except ValueError:
                return val

        # --- 1. Iterate through JSON to populate structures ---
        for g in groups:
            g_seq = g['group_seq']
            g_name = g['group_name']

            # Basic Input
            basic_rows.append({
                'Group seq': parse_val(g_seq),
                'Group name': g_name,
                'Hab area (proportion)': parse_val(g.get('habitat_area')),
                'Biomass in habitat area (t/km^2)': parse_val(g.get('biomass_habitat_area')),
                'Production / biomass (/year)': parse_val(g.get('pb')),
                'Consumption / biomass (/year)': parse_val(g.get('qb')),
                'Ecotrophic Efficiency': parse_val(g.get('ee')),
                'Other mortality': parse_val(g.get('other_mort')),
                'Unassim. consumption': parse_val(g.get('gs')),
                'Detritus import (t/km^2/year)': parse_val(g.get('detritus_import'))
            })

            # Export (Represents Landings + Discards)
            exp_val = float(g.get('export', 0))
            if exp_val > 0:
                has_export = True
            landings_rows.append({'Group seq': g_seq, 'Group name': g_name, 'Total': parse_val(g.get('export', 0))})

            # Taxonomy
            td = g.get('taxon_descr')
            if td and td != "None":
                has_tax = True
                tax_rows.append({'Group seq': g_seq, 'Group name': g_name, 'Taxon Description': td})

            # TL (If available in the extended JSON)
            tl = g.get('tl')
            if tl and tl != "-9999":
                has_tl = True
                tl_rows.append({'Group seq': g_seq, 'Group name': g_name, 'TL': parse_val(tl)})

            # Detritus identification
            if g.get('pp') == "2":
                detritus_pools.add(g_seq)

            # Diet & Detritus Fate Mapping
            import_val = float(g.get('diet_imp', 0))
            if import_val > 0:
                diet_matrix.at['Import', g_seq] = import_val

            diet_info = g.get('diet_descr')
            if diet_info and diet_info.get('diet'):
                d_list = diet_info['diet']
                if isinstance(d_list, dict):  # Handle case where there's only one prey item
                    d_list = [d_list]

                for item in d_list:
                    prey = str(item['prey_seq'])
                    prop = float(item.get('proportion', 0))
                    fate = float(item.get('detritus_fate', 0))

                    # Assign Diet
                    if prop > 0 and prey in diet_matrix.index:
                        diet_matrix.at[prey, g_seq] = prop

                    # Assign Detritus Fate Routing
                    if fate > 0:
                        detritus_pools.add(prey)
                        if prey in det_fate_matrix.columns:
                            det_fate_matrix.at[g_seq, prey] = fate

        # --- 2. Build and Save DataFrames to Excel ---
        self.logger.info(f"Writing reconstructed files to: {output_excel}")
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:

            # 0. Metadata
            meta_dict = self._parse_metadata_from_filename(json_path)
            df_meta = pd.DataFrame(list(meta_dict.items()), columns=['Key', 'Value'])
            df_meta.to_excel(writer, sheet_name='Metadata', index=False, header=False)

            # 1. Basic input (Mandatory)
            df_basic = pd.DataFrame(basic_rows)
            df_basic.to_excel(writer, sheet_name='Basic input', index=False)

            # 2. Diet composition (Mandatory)
            df_diet = diet_matrix.copy()
            # Add Prey Names as the first column to match EwE format
            df_diet.insert(0, 'Prey \\ Predator', [group_names.get(str(idx), idx) for idx in df_diet.index])

            # Add a Sum row at the bottom
            df_diet.loc['Sum'] = df_diet.sum(numeric_only=True)
            df_diet.at['Sum', 'Prey \\ Predator'] = 'Sum'

            # Convert index back to column for output
            df_diet.reset_index(inplace=True)
            df_diet.rename(columns={'index': 'Prey ID'}, inplace=True)
            df_diet.to_excel(writer, sheet_name='Diet composition', index=False)

            # 4. Landings (Using the consolidated Export value)
            if has_export:
                df_landings = pd.DataFrame(landings_rows)
                # Filter out rows that are entirely 0 or empty to keep it clean
                df_landings = df_landings[df_landings['Total'] != ""]
                df_landings = df_landings[df_landings['Total'] > 0]
                df_landings.to_excel(writer, sheet_name='Landings', index=False)
                self.logger.info("  -> Included 'Landings' sheet (calculated from 'export' param)")

            # 5. Taxonomy
            if has_tax:
                df_tax = pd.DataFrame(tax_rows)
                df_tax.to_excel(writer, sheet_name='Taxonomy', index=False)
                self.logger.info("  -> Included 'Taxonomy' sheet")

            # 6. TL
            if has_tl:
                df_tl = pd.DataFrame(tl_rows)
                df_tl.to_excel(writer, sheet_name='TL', index=False)
                self.logger.info("  -> Included 'TL' sheet")

            # 7. Detritus Fate
            if detritus_pools:
                det_cols = list(detritus_pools)
                df_det = det_fate_matrix[det_cols].copy()

                # Map column IDs back to names for readability
                df_det.columns = [group_names.get(str(c), c) for c in df_det.columns]

                # Insert Source Info
                df_det.insert(0, 'Source Name', [group_names.get(str(idx), idx) for idx in df_det.index])
                df_det.insert(0, 'Source ID', df_det.index)

                # Remove rows where a source routes 0.0 to all detritus pools (Living groups with no routing)
                # Actually, standard EwE format keeps all groups in the rows, so we'll leave them in for structural consistency
                df_det.to_excel(writer, sheet_name='Detritus fate', index=False)
                self.logger.info(f"  -> Included 'Detritus fate' sheet (Pools identified: {len(detritus_pools)})")

        self.logger.info("\nSuccess! Validation file generated.")
        return output_excel


def main():
    parser = argparse.ArgumentParser(
        description="EwE Model Data Converter",
        epilog="""
Usage Examples:
  Option 1: Convert directory to JSON, then generate Excel from that JSON:
      python ewe_converter.py -d ./data/13_north_Formats

  Option 2: Generate Excel directly from an existing JSON file:
      python ewe_converter.py -j ./data/13_north_Formats/my_model.json
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--dir', type=str, help='Path to directory containing CSV/Excel EwE inputs.')
    group.add_argument('-j', '--json', type=str, help='Path to an existing JSON model file.')

    ALL_INPUTS = [
        "..\\13_Humboldt_Current\\13_north_(2018)\\13_north_(2018)_extracted\\",
        "..\\13_Humboldt_Current\\13_south_(2026)\\13_south_(2026)_extracted\\",
        "..\\36_South_China_Sea\\36_north_(2007)\\36_north_(2007)_extracted\\1970s\\",
        "..\\36_South_China_Sea\\36_north_(2007)\\36_north_(2007)_extracted\\2000s\\",
        "..\\47_East_China_Sea\\47_(2022)_extracted\\M1997\\",
        "..\\47_East_China_Sea\\47_(2022)_extracted\\M2018\\",
        "..\\48_Yellow_Sea\\48_reefs_(2022)\\48_reefs_(2022)_extracted\\artificial_reefs\\",
        "..\\48_Yellow_Sea\\48_reefs_(2022)\\48_reefs_(2022)_extracted\\neutral_reefs\\",
        "..\\48_Yellow_Sea\\48_southwestern_(2022)\\48_southwestern_(2022)_extracted\\",
        "..\\49_Kuroshio_Current\\49_(2019)_extracted\\"
    ]

    args = parser.parse_args()

    converter = EwEConverter()

    if args.dir:
        input_directory = os.path.abspath(args.dir)
        if not os.path.exists(input_directory) or not os.path.isdir(input_directory):
            print(f"Error: Directory '{input_directory}' does not exist.")
            return

        for input_dir in ALL_INPUTS:
            print(f"\n--- Running on", input_dir)
            json_file = converter.csv_to_json(input_dir)
            if json_file:
                converter.json_to_excel(json_file)

        # print(f"\n--- Running Pipeline Option 1 (Dir -> JSON -> Excel) ---")
        # json_file = converter.csv_to_json(input_directory)
        # if json_file:
        #     converter.json_to_excel(json_file)

    elif args.json:
        json_path = os.path.abspath(args.json)
        if not os.path.exists(json_path) or not os.path.isfile(json_path):
            print(f"Error: JSON file '{json_path}' does not exist.")
            return

        print(f"\n--- Running Pipeline Option 2 (JSON -> Excel) ---")
        converter.json_to_excel(json_path)


if __name__ == "__main__":
    main()