from app.common import *
from simulation import FactorStress
from app.ui.utils_ui import render_info_box
from app.cached_functions import cached_monte_carlo, cached_historical
import streamlit as st
import time


#========= SIMULATION UI FUNCTIONS =========

def render_scenario(portfolios:dict, name:str):
    """
    Renders the scenario builder.
    Displays locked state with previously selected values after confirmation.
    """
    scenario_name = st.session_state.current_scenario
    scenarios = portfolios[name]['scenarios']

    #New scenario creation
    if scenario_name is None or scenario_name not in scenarios:
        scenario_placeholder_name = f"Scenario {portfolios[name]['scenario_ctr']}"
        scenario_name_input = st.text_input("**Input Scenario Name:**", placeholder=scenario_placeholder_name, max_chars=30)
        st.caption("_This will allow you to distingiuish between multiple risk scenarios within the same portfolio._")

        sim_method = st.radio("**Simulation Method:**", ['Monte Carlo', 'Historical Replay'], horizontal=True, index=0)
        st.session_state.current_sim_method = sim_method
        sims = T = regime = level = crisis = None

        if sim_method == 'Monte Carlo':
            #Initialize # of days and # of sims
            mccol1, mccol2 = st.columns(2)
            with mccol1:
                sims = st.number_input("Number of simulations:", min_value=100, max_value=5000, value=1000, step=100)
            with mccol2:
                T = st.number_input("Number of days:", min_value=25, max_value=1000, value=250, step=25)
            if sims >= 3000:
                st.warning("The large amount of simulations may take longer to compute", icon="⚠️")

            #Input risk profile
            regime_options = ['Low', 'Medium', 'High']
            regime = st.selectbox("**Current State of the Market:**", regime_options)
            level_options = ['1.0x', '1.15x', '1.25x', '1.35x', '1.5x']
            level = st.selectbox("**Additional Stress Multiplier:**", level_options)
            st.caption("_Stress multiplier allows you to simulate extreme tail events beyond historical volatility regimes_.")
        
        if sim_method == 'Historical Replay':
            #Choose crisis event
            crisis_options = {'Dot-Com Bubble': 'DOT-COM',
                              '2008 Global Finanical Crisis': '2008 GFC',
                              '2011 Euro Zone Crisis': '2011 Euro',
                              'COVID-19': 'COVID',
                              '2022 Inflation Crash': '2022 Inf'}
            
            crisis_expanded = st.selectbox("**Select Crisis Event:**", list(crisis_options.keys()))
            crisis = crisis_options[crisis_expanded]
        
        final_scenario_name = scenario_name_input if scenario_name_input else scenario_placeholder_name
        return sim_method, sims, T, regime, level, crisis, final_scenario_name

    else:
        #Display locked state for existing scenario
        scenario_name = st.text_input("**Input Scenario Name:**", value=scenario_name, disabled=True)
        st.caption("_This will allow you to distingiuish between multiple risk scenarios within the same portfolio._")

        #Locked Monte Carlo inputs
        if scenarios[scenario_name]['sim_method'] == 'Monte Carlo':
            st.radio("**Simulation Method:**", ['Monte Carlo', 'Historical Replay'],
                      horizontal=True, 
                      index=0, 
                      disabled=True)
            
            mccol1, mccol2 = st.columns(2)
            with mccol1:
                st.number_input("Number of simulations:", 
                                       value=scenarios[scenario_name]['# sims'], 
                                       disabled=True)
            with mccol2:
                st.number_input("Number of days:",
                                value=scenarios[scenario_name]['# days'],
                                disabled=True)
            
            regime_options = ['Low','Medium','High']
            regime_display_idx = regime_options.index(scenarios[scenario_name]['regime'])
            st.selectbox("**Current State of the Market:**", 
                     regime_options, 
                     index=regime_display_idx,
                     disabled=True)
            
            level_options = ['1.0x', '1.15x', '1.25x', '1.35x', '1.5x']
            level_display_idx = level_options.index(scenarios[scenario_name]['level'])
            st.selectbox("**Additional Stress Multiplier:**", 
                     level_options,
                     index=level_display_idx,
                     disabled=True)
            st.caption("_Stress multiplier allows you to simulate extreme tail events beyond historical volatility regimes_.")
                
        #Locked Historical Replay inputs
        elif scenarios[scenario_name]['sim_method'] == 'Historical Replay':
            st.radio("**Simulation Method:**", ['Monte Carlo', 'Historical Replay'],
                     horizontal=True, 
                      index=1, 
                      disabled=True)  

            crisis_options = {'DOT-COM': 'Dot-Com Bubble',
                              '2008 GFC':'2008 Global Finanical Crisis',
                              '2011 Euro':'2011 Euro Zone Crisis',
                              'COVID':'COVID-19',
                              '2022 Inf':'2022 Inflation Crash'}
            
            crisis_display = crisis_options.get(scenarios[scenario_name]['crisis'], scenarios[scenario_name]['crisis'])
            st.selectbox("**Select Crisis Event:**", 
                         options=[crisis_display],
                         index=0,
                         disabled=True)

        return None, None, None, None, None, None, None

                    
def render_classification(portfolios:dict, name:str):
    """ 
    Renders the classification of each asset in the portfolio based on the factors chosen.
    Displays locked state with previously selected values after confirmation.
    """
    scenario_name = st.session_state.current_scenario
    scenarios = portfolios[name]['scenarios']

    #Check if there is enough data or not any incomplete tickers to render classification
    if len(portfolios[name]['configs']['df']['data']) < 70: 
        render_info_box("Classification requires more historical data", icon='⚠️', margin_top='48px', bg='#3D3B15', txt="#FCF4BA", border="#1D1D14")
        return None
    if  portfolios[name]['configs']['df']['warning'] is not None:
        render_info_box("Classification unavailable due to incomplete ticker data", icon='⚠️', margin_top='48px', bg='#3D3B15', txt="#FCF4BA", border="#1D1D14")
        return None
    
    #Derive current sim method from unconfirmed scenario
    if scenario_name is None or scenario_name not in scenarios:
        sim_method = st.session_state.current_sim_method
        
        if sim_method == 'Monte Carlo':
            classifyq = st.checkbox("**Classify your portfolio?**")
            if not classifyq:
                render_info_box("Classification allows you to understand how your portfolio is exposed to different risk factors.", 
                                icon='🏷️', max_width='600px', margin_top='48px')
            else:
                factors_expanded = ['3-factor Fama-French',
                                    '5-factor Fama-French',
                                    'Market Premium',
                                    'Size Premium',
                                    'Value Premium',
                                    'Profitability Factor',
                                    'Investment Factor',
                                    'Momentum Effect']
                factors_choice = st.multiselect("Select a Model or Individual Factors:", factors_expanded)
                                
                factor_condensed = ['FF3','FF5','Mkt-RF','SMB','HML','RMW','CMA','Mom']

                factor_convert = {e: c for e, c in zip(factors_expanded, factor_condensed)}               
                factors = [factor_convert[f] for f in factors_choice]

                return factors
        else:
            render_info_box("Classification inputs are only available for Monte Carlo scenarios.", 
                            icon='🏷️', margin_top='32px')
    
    #Display locked state if scenario exists
    elif scenarios[scenario_name]['sim_method'] == 'Monte Carlo' and scenarios[scenario_name]['factors']:
        st.checkbox("**Classify your portfolio?**", value=True, disabled=True)

        factors_expanded = ['3-factor Fama-French',
                            '5-factor Fama-French',
                            'Market Premium',
                            'Size Premium',
                            'Value Premium',
                            'Profitability Factor',
                            'Investment Factor',
                            'Momentum Effect']
        
        factors_condensed = ['FF3', 'FF5', 'Mkt-RF', 'SMB','HML','RMW','CMA','Mom']

        factor_convert = {c: e for c, e in zip(factors_condensed, factors_expanded)}
        expanded_chosen_f = [factor_convert[f] for f in scenarios[scenario_name]['factors']]

        st.multiselect("Select a Model or Individual Factors:", 
                        options=factors_expanded,
                        default=expanded_chosen_f,
                        disabled=True)
        
        return scenarios[scenario_name]['factors']
    
    else:
        if scenarios[scenario_name]['sim_method'] == 'Monte Carlo':
            st.checkbox("**Classify your portfolio?**", value=False, disabled=True)
            render_info_box("Classification allows you to understand how your portfolio is exposed to different risk factors.", 
                            icon = '🏷️', max_width='600px', margin_top='48px')
        else:
            render_info_box("Classification inputs are only available for Monte Carlo scenarios.", 
                            icon='🏷️', margin_top='52px')

        return None


def render_shocks(portfolios:dict, name:str, factors:list):
    """
    Renders factor shock sliders.
    Displays locked state with previously selected values after confirmation.
    """
    scenario_name = st.session_state.current_scenario
    scenarios = portfolios[name]['scenarios']

    if scenario_name is None or scenario_name not in scenarios:
        shock_ask = st.checkbox("**Would you like to induce factor-stress into the simulation?**")
        st.caption("_FF3/FF5 apply uniform stress. Select individual factors for asymmetric shocks._")

        if shock_ask:
            shock_dict = {}
            shock_range = range(-50, 51, 5)
            for f in factors:
                percents = [f'{s}%' for s in shock_range]
                shock_dict[f] = st.select_slider(f"{f} Shock", key=f'shock_{f}', options=percents, value='0%')
                
            shocks_dict = {key: (float(value.replace('%', ''))/100) for key,value in shock_dict.items()}

            return shocks_dict
        
        #No shocks desired
        else:   
            render_info_box("No factor shocks haven been applied to this scenario.", icon='⚡', margin_top='-40px')

            return {}
            
    #Display locked state if scenario already exists
    else:
        #Check if factor shocks exist in the scenario
        has_shocks = 'factor_shocks' in scenarios[scenario_name] and scenarios[scenario_name]['factor_shocks']

        if has_shocks:
            st.checkbox("**Would you like to induce factor-stress into the simulation?**", value=True, disabled=True)
            st.caption("_FF3/FF5 apply uniform stress. Select individual factors for asymmetric shocks._")
            
            fs = scenarios[scenario_name]['factor_shocks']
            fs_copy = scenarios[scenario_name]['factor_shocks'].copy()
            shock_range = range(-50, 51, 5)
            percents = [f'{s}%' for s in shock_range]

            def uniform_check(mapping, group):
                """Checks if all values in a group are the same in a mapping (FF3, FF5)"""
                values = [mapping[f] for f in group]
                return len(set(values)) == 1
            
            FF3_factors = ['Mkt-RF', 'SMB', 'HML']
            FF5_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
            #Show FF3 instead of individual factors if applicable
            if set(FF3_factors).issubset(fs.keys()) and uniform_check(fs, FF3_factors) and not any(f in fs for f in ['RMW', 'CMA']):
                shock = fs['Mkt-RF']
                st.select_slider('FF3 Shock',
                                value=f'{int(shock*100)}%',
                                options=percents,
                                disabled=True)
                for factor in FF3_factors:
                    del fs_copy[factor]

            #Show FF5 instead of individual factors if applicable
            elif set(FF5_factors).issubset(fs.keys()) and uniform_check(fs, FF5_factors):
                shock = fs['Mkt-RF']
                st.select_slider('FF5 Shock',
                                 value=f'{int(shock*100)}%',
                                 options=percents,
                                 disabled=True)
                for factor in FF5_factors:
                    del fs_copy[factor]

            for factor, shock_value in fs_copy.items():
                shock_percentage = f'{int(round(shock_value*100))}%'
                st.select_slider(f'{factor} Shock', 
                                 value=shock_percentage, 
                                 options=percents, 
                                 disabled=True)

            return scenarios[scenario_name]['factor_shocks']
    
        else:
            st.checkbox("**Would you like to induce factor-stress into the simulation?**", value=False, disabled=True)
            st.caption("_FF3/FF5 apply uniform stress. Select individual factors for asymmetric shocks._")
            render_info_box("No factor shocks haven been applied to this scenario.", icon='⚡', margin_top='-40px')

            return {}


def simulation_config():
    """ 
    UI for Simulation Configuration page.
    """
    st.header("Simulation Configuration", divider=True)
    portfolios = st.session_state.portfolios
    name = st.session_state.current_portfolio
    scenarios = portfolios[name]['scenarios']
    scenario_name = st.session_state.current_scenario

    scol1, scol2 = st.columns(2)

    with scol1:
        with st.container(border=True, height=610):
            st.markdown("#### Scenario Builder")
            result = render_scenario(portfolios, name)
            if result:
                sim_method, sims, T, regime, level, crisis, final_scenario_name = result
                factors = s_means = s_means_warning = None

    with scol2:
        with st.container(border=True, height=610):
            st.markdown("#### Classification & Factor Shocks")
            factors = render_classification(portfolios, name)

            shocks = None
            if factors is not None:
                df = portfolios[name]['configs']['df']['data']
                f = FactorStress(df)
                
                #Run factor functions
                _, process_error = f.process_factors(factors)
                classify_df, classify_error = f.classify_factors()

                if process_error:
                    st.error(f"Error processing factors: {process_error}", icon="❌")
                elif classify_error:
                    st.error(f"Error classifying factors: {classify_error}", icon="❌")
                else:
                    #Display classification
                    if not classify_df.empty:
                        st.dataframe(classify_df, width='stretch')
                    shocks = render_shocks(portfolios, name, factors)
                    if shocks:
                        s_means, s_means_error, s_means_warning = f.stress_means(shocks)
                        if s_means_error:
                            st.error(f"Error in stressing means: {s_means_error}", icon="❌")
                        elif s_means_warning:
                            st.warning(s_means_warning, icon="⚠️")

    #Confirmation Button
    if final_scenario_name in scenarios or st.session_state.current_scenario in scenarios:
        st.button("Confirm Scenario", width='stretch', disabled=True)
    else:
        confirm = st.button("Confirm Scenario", width='stretch')
        if confirm:
            with st.spinner("Running simulation..."):
                try:
                    df = portfolios[name]['configs']['df']['data']
                    weights = portfolios[name]['configs']['weights']
                    
                    if sim_method == 'Monte Carlo':
                        #Cached function requires hashable inputs
                        results, error = cached_monte_carlo(
                            T, sims, weights,
                            df.values.tolist(),
                            df.index.tolist(),
                            df.columns.tolist(),
                            regime, level, s_means
                        )
                    elif sim_method == 'Historical Replay':
                        results, error = cached_historical(
                            df.values.tolist(),
                            df.index.tolist(),
                            df.columns.tolist(),
                            crisis
                        )
                    
                    if error:
                        st.error(f"Simulation error: {error}", icon='❌')
                    else:
                        #Store in session state
                        if sim_method == 'Monte Carlo':
                                scenarios[final_scenario_name] = {
                                    'sim_method': sim_method,
                                    '# sims': sims,
                                    '# days': T,
                                    'regime': regime,
                                    'level': level,
                                    'factors': factors,
                                    'factor_shocks': shocks,
                                    'stressed_means': {'means': s_means,
                                                       'warning': s_means_warning},
                                    'results': results,
                                    'display_paths': None
                                    }
                                
                        elif sim_method == 'Historical Replay':
                            scenarios[final_scenario_name] = {
                                'sim_method': sim_method,
                                'crisis': crisis,
                                'results': results
                                }
                            
                        portfolios[name]['scenario_ctr'] += 1
                        portfolios[name]['sim_complete'] = True
                        st.session_state.current_scenario = final_scenario_name
                        st.success("Scenario successfully created!", icon="✅")
                        time.sleep(1)
                        st.rerun()

                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}", icon='❌')

    #Create another scenario
    scenario_create = st.button("Create Another Scenario?", width='stretch')
    if scenario_create:
        st.session_state.current_scenario = None
        st.success("Scenario sucessfully initialized!", icon="✅")
        st.rerun()

    #Scenario selectbox
    scenarios = list(portfolios[name]['scenarios'].keys())
    selected_scenario = st.selectbox("**View Previous Scenarios**", 
                                    options=["Select a scenario..."] + scenarios,
                                    index=0 if st.session_state.current_scenario is None or st.session_state.current_scenario not in scenarios else scenarios.index(st.session_state.current_scenario) + 1)
            
    #Revisit old scenario
    if selected_scenario != "Select a scenario..." and selected_scenario != st.session_state.current_scenario:
        st.session_state.current_scenario = selected_scenario
        st.rerun()