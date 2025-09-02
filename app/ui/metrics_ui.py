from app.common import * 
from metrics import calculate_metrics, plot_cumulative_returns, plot_PCR, SimulationAnalyzer
from app.ui.utils_ui import render_info_box
from app.cached_functions import cached_calculate_metrics, cached_all_MC_metrics
import pandas as pd
import streamlit as st


#========= RESULTS & METRICS FUNCTIONS =========

def render_metric_cards(portfolios:dict, name:str):
    """
    Renders metric cards for a select scenario with comparision to baseline metrics.
    """
    scenarios = portfolios[name]['scenarios']

    if ('current_scenario' not in st.session_state or st.session_state.current_scenario not in scenarios):
        st.session_state.current_scenario = list(scenarios.keys())[0]
    scenario_name = st.session_state.current_scenario

    weights = portfolios[name]['configs']['weights']
    baseline_df = portfolios[name]['configs']['df']['data']
    scenario_labels = []

    #Get metrics
    baseline_result, baseline_error = cached_calculate_metrics(weights,
                                                               baseline_df.values.tolist(),
                                                               baseline_df.index.tolist(),
                                                               baseline_df.columns.tolist())
    if baseline_error:
        st.error(f"Error calculating baseline metrics: {baseline_error}", icon='❌')
        return
    baseline_metrics = baseline_result[0]
    portfolios[name]['metrics'] = baseline_result
 
    if st.session_state.sim_analyzer is None or not portfolios[name]['sim_analyzer']['init']:
        S = SimulationAnalyzer()
        S.weights = weights
        st.session_state.sim_analyzer = S
        portfolios[name]['sim_analyzer']['init'] = True
        portfolios[name]['sim_analyzer']['object'] = S
    else:
        S = portfolios[name]['sim_analyzer']['object'] 
        st.session_state.sim_analyzer = S

    for label, data in scenarios.items():
        scenario_labels.append(label)
        if label not in S.mc_batches and label not in S.hist_batches:
            sim_returns = data['results']
            sim_type = data['sim_method']
            hist_batches, add_error = S.add_simulation(label, sim_type, sim_returns) #add_simulation returns historical replay results
            if add_error:
                st.error(f"Error adding simulation {label}: {add_error}", icon='❌')
                return
    
    #Gather all Monte Carlo metrics for each sim and combine all metrics across all sims
    if S.mc_batches:
        mc_results, mc_error = cached_all_MC_metrics(S.mc_batches, weights)
        if mc_error:
            st.error(f"Error calculating Monte Carlo metrics: {mc_error}", icon='❌')
        else:
            S.all_mc_metrics = mc_results[0]
            S.all_mc_PCR = mc_results[1]
            S.all_mc_cum_returns = mc_results[2]
        
    combined_results, combined_error = S.get_combined_metrics(weights)
    if combined_error:
        st.error(f"Error getting combined metrics: {combined_error}", icon='❌')
        return 
    combined_metrics = combined_results[0]

    current_sim_type = scenarios[scenario_name]['sim_method']
    if current_sim_type == 'Monte Carlo': #Monte Carlo case
        #Select scenario
        scol1, scol2 = st.columns(2)
        with scol1:
            select_scenario = st.selectbox("**Select Scenario to View:**", scenario_labels, index=scenario_labels.index(scenario_name))
            #Check if selection changed and update session state
            if select_scenario != scenario_name:
                st.session_state.current_scenario = select_scenario
                st.rerun()
        with scol2:
            path_type = st.selectbox("**Select Path to View:**", ['Representative','Best', 'Worst'], index=0)
            if path_type != st.session_state.get('current_path_type', 'Representative'):
                st.session_state.current_path_type = path_type
                st.rerun()
            
        #Populate all scenarios with each path
        scenario_name = st.session_state.current_scenario
        current_path = st.session_state.current_path_type
        scenario_labels = portfolios[name]['scenarios'].keys()
        for s_name in scenario_labels:
            if scenarios[s_name]['display_paths'] is None:
                repr_results, repr_error = S.get_display_path(s_name, 'Representative')
                best_results, best_error = S.get_display_path(s_name, 'Best')
                worst_results, worst_error = S.get_display_path(s_name, 'Worst')
                if repr_error or best_error or worst_error:
                    st.error(f"Error getting display path: {repr_error or best_error or worst_error}", icon='❌')
                    return 
                
                scenarios[s_name]['display_paths'] = {
                    'Representative': repr_results,
                    'Best': best_results,
                    'Worst': worst_results
                }

        #Extract current scenario metrics for current use
        metrics = scenarios[scenario_name]['display_paths'][current_path][0]

    else: #Historical Replay case
        select_scenario = st.selectbox("**Select Scenario to View:**", scenario_labels, index=scenario_labels.index(scenario_name))
        # Check if selection changed and update session state
        if select_scenario != scenario_name:
            st.session_state.current_scenario = select_scenario
            st.rerun()

        #Use the current session state value instead of the old scenario_name
        scenario_name = st.session_state.current_scenario
        if scenario_name in combined_metrics:
            metrics = combined_metrics[scenario_name].loc['Portfolio']
        else:
            st.error(f"Scenario {scenario_name} not found in combined metrics", icon='❌')

    #Display metric cards
    default_metrics = ['Annual Volatility', 'Expected Return', 'Sharpe Ratio', '95% VaR', 'Max Drawdown']
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)

    with mcol1: #Annual Volatility
        metric_val = metrics[default_metrics[0]].item()
        baseline_val = baseline_metrics[default_metrics[0]].item()
        vol_delta = metric_val - baseline_val
        st.metric(label=default_metrics[0], value=f"{metric_val:.4f}", delta=f"{vol_delta:.4f}", delta_color='inverse', border=True)
        
    with mcol2: #Expected Return
        metric_val = metrics[default_metrics[1]].item()
        baseline_val = baseline_metrics[default_metrics[1]].item()
        vol_delta = metric_val - baseline_val
        st.metric(label=default_metrics[1], value=f"{metric_val:.4f}", delta=f"{vol_delta:.4f}", border=True)

    with mcol3: #Sharpe Ratio
        metric_val = metrics[default_metrics[2]].item()
        baseline_val = baseline_metrics[default_metrics[2]].item()
        sharpe_delta = metric_val - baseline_val
        st.metric(default_metrics[2], f"{metric_val:.4f}", f"{sharpe_delta:.4f}", border=True)
        
    with mcol4: #95% VaR
        metric_val = metrics[default_metrics[3]].item()
        baseline_val = baseline_metrics[default_metrics[3]].item()
        VaR_delta = metric_val - baseline_val
        st.metric(default_metrics[3], f"{metric_val:.4f}", f"{VaR_delta:.4f}", border=True)
        
    with mcol5: #Max Drawdown
        metric_val = metrics[default_metrics[4]].item()
        baseline_val = baseline_metrics[default_metrics[4]].item()
        mdd_delta = metric_val - baseline_val
        st.metric(default_metrics[4], f"{metric_val:.4f}", f"{mdd_delta:.4f}", border=True)


def render_divider(portfolios:dict, name:str):
    """ 
    Renders info captions for metrics and final portfolio value.
    """
    initial_value = portfolios[name]['configs']['portfolio_value']
    scenarios = portfolios[name]['scenarios']
    current_path = st.session_state.current_path_type
    sim_analyzer = st.session_state.sim_analyzer

    if ('current_scenario' not in st.session_state or st.session_state.current_scenario not in scenarios):
        st.session_state.current_scenario = list(scenarios.keys())[0]
    scenario_name = st.session_state.current_scenario

    dcol1, dcol2 = st.columns(2)
    with dcol1: #Final portfolio value
        if scenarios[scenario_name]['sim_method'] == 'Monte Carlo':
            cum_returns = scenarios[scenario_name]['display_paths'][current_path][2]
        elif scenarios[scenario_name]['sim_method'] == 'Historical Replay':
            cum_returns = sim_analyzer.combined_cum_returns[scenario_name]

        final_portfolio_value = initial_value*cum_returns.iloc[-1]
        PnL = final_portfolio_value - initial_value
        if PnL >= 0:
            color = '#00c851'
        else:
            color = '#ff4b4b'

        st.markdown(f"""
        <div style="
            background-color: rgba(255, 255, 255, 0.05);
            padding: 15px 20px;
            border-radius: 6px;
            border-left: 3px solid {color};
            margin: 10px 0;
        ">
            <strong>Final Portfolio Value:</strong> 
            <span style="color: {color}; font-size: 20px; font-weight: bold;">
                ${final_portfolio_value:,.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        with dcol2: #Info captions
            st.markdown("""
            • **Current metrics are compared to original portfolio data as a baseline**

            • **Representative Monte Carlo path is the "highest probable" path**
            
            • **Best and Worst Monte Carlo path are based on Sharpe Ratio**
            """)


def render_cumulative_returns(portfolios:dict, name:str):
    """
    Renders final portfolio value and cumulative returns line chart.
    """
    initial_value = portfolios[name]['configs']['portfolio_value']
    scenarios = portfolios[name]['scenarios']

    if ('current_scenario' not in st.session_state or st.session_state.current_scenario not in scenarios):
        st.session_state.current_scenario = list(scenarios.keys())[0]
    scenario_name = st.session_state.current_scenario

    current_path = st.session_state.current_path_type
    sim_analyzer = st.session_state.sim_analyzer

    has_multiple_scenarios = len(scenarios.keys()) > 1

    #Get combined cumulative returns 
    combined_cum_returns = sim_analyzer.combined_cum_returns

    #Monte Carlo scenario
    if scenarios[scenario_name]['sim_method'] == 'Monte Carlo':
        if has_multiple_scenarios:
            view_mode = st.radio("**View Mode:**", ['Detailed View', 'Compare Scenarios'], horizontal=True, 
                                 index=0 if st.session_state.view_mode == 'Detailed View' else 1)
        else:
            view_mode = st.radio("**View Mode:**", ['Detailed View', 'Compare Scenarios'], horizontal=True, index=0, disabled=True)

    elif scenarios[scenario_name]['sim_method'] == 'Historical Replay': #Historical Replay scenario
        view_mode = st.radio("**View Mode:**", ['Detailed View', 'Compare Scenarios'], horizontal=True, index=1, disabled=True)

    st.session_state.view_mode = view_mode

    if view_mode == 'Detailed View': #Shows all three path cumulative returns for selected scenario
        st.caption("_Plots representative, best, and worst path of current Monte Carlo scenario_")
        repr_results = scenarios[scenario_name]['display_paths']['Representative']
        best_results = scenarios[scenario_name]['display_paths']['Best']
        worst_results = scenarios[scenario_name]['display_paths']['Worst']

        #Cumulative returns
        repr_cum_returns, best_cum_returns, worst_cum_returns = repr_results[2], best_results[2], worst_results[2]
        cum_returns_df = pd.concat([repr_cum_returns, best_cum_returns, worst_cum_returns], axis=1)
        cum_returns_df.columns = ['Representative', 'Best', 'Worst']

        #Plot
        plot_cum, plot_cum_error = plot_cumulative_returns(initial_value, view_mode, cum_returns_df, current_scenario=scenario_name, current_path=current_path)
        if plot_cum_error:
            st.error(f"Error plotting cumulative returns: {plot_cum_error}", icon='❌')
            return
        st.plotly_chart(plot_cum, width='stretch')
    
    elif view_mode == 'Compare Scenarios': #Compares representative paths of all scenarios
        st.caption('_Compares representative path of Monte Carlo scenarios to Historical Replay scenarios_')
        all_cum_returns = {}
        for label in scenarios.keys():
            if scenarios[label]['sim_method'] == 'Monte Carlo':
                if scenarios[label]['display_paths']['Representative'] is not None:
                    repr_results = scenarios[label]['display_paths']['Representative']
                else:
                    repr_results, repr_error = sim_analyzer.get_display_path(label, 'Representative')
                    if repr_error:
                        st.error(f"Error getting display path: {repr_error}", icon='❌')
                        return
                all_cum_returns[label] = repr_results[2]
                
            elif scenarios[label]['sim_method'] == 'Historical Replay':
                all_cum_returns[label] = combined_cum_returns[label].iloc[0]

        all_repr_df = pd.DataFrame(all_cum_returns)
        
        #Plot
        plot_cum, plot_cum_error = plot_cumulative_returns(initial_value, view_mode, all_repr_df, current_scenario=scenario_name)
        if plot_cum_error:
            st.error(f"Error plotting cumulative returns: {plot_cum_error}", icon='❌')
            return
        st.plotly_chart(plot_cum, width='stretch')


def render_PCR(portfolios:dict, name:str):
    """ 
    Renders Percent Contribution of Risk bar chart overlaid with baseline PCR.
    """
    scenarios = portfolios[name]['scenarios']

    if ('current_scenario' not in st.session_state or st.session_state.current_scenario not in scenarios):
        st.session_state.current_scenario = list(scenarios.keys())[0]
    scenario_name = st.session_state.current_scenario

    sim_analyzer = st.session_state.sim_analyzer
    weights = portfolios[name]['configs']['weights']
    baseline_df = portfolios[name]['configs']['df']['data']

    #Get baseline PCR dataframe
    baseline_result, baseline_error = calculate_metrics(weights, baseline_df)
    if baseline_error:
        st.error(f"Error calculating baseline metrics: {baseline_error}", icon='❌')
        return
    baseline_PCR = baseline_result[1]
    
    combined_PCR = sim_analyzer.combined_PCR

    #Get current PCR dataframe
    if scenarios[scenario_name]['sim_method'] == 'Monte Carlo':
        current_PCR = sim_analyzer.all_mc_PCR[scenario_name].mean()
        current_PCR = pd.DataFrame({'Avg PCR': current_PCR}).T
        st.caption("_Monte Carlo PCR is computed by averaging every paths individual PCR_")
    else:
        current_PCR = combined_PCR[scenario_name]
        st.caption("_Historical Replay PCR only takes into account the crisis period_")
    #Plot
    plot_pcr, plot_pcr_error = plot_PCR(current_PCR, baseline_PCR)
    if plot_pcr_error:
        st.error(f"Error plotting PCR: {plot_pcr_error}", icon='❌')
        return
    st.plotly_chart(plot_pcr, width='stretch')


def render_visualize_metrics():
    """ 
    Renders KDEs comparing each metric across scenarios.
    """
    sim_analyzer = st.session_state.sim_analyzer

    with st.container(border=True):
        st.markdown("#### KDE Comparison of Portfolio Metrics Across Simulations")
        
        if len(sim_analyzer.mc_batches.keys()) == 0:
            render_info_box("KDE plots require at least one Monte Carlo simulation to display.", icon='📊')
        else:
            default_metrics = ['Annual Volatility', 'Expected Return', 'Sharpe Ratio', '95% VaR', 'Max Drawdown']
            metric_choice = st.selectbox("**Choose Metric KDE to Display**:", default_metrics)

            plot_metrics, plot_metrics_error = sim_analyzer.visualize_metrics(metric_choice)
            if plot_metrics_error:
                st.error(f"Error in plotting metric KDEs: {plot_metrics_error}")
                return
            else:
                st.plotly_chart(plot_metrics, width='stretch')


def metrics():
    """
    UI for Metrics & Visualization page.
    """
    portfolios = st.session_state.portfolios
    name = st.session_state.current_portfolio
    scenarios = portfolios[name]['scenarios']
    scenario_name = st.session_state.current_scenario
    st.header("Metrics & Visualization", divider=True)

    render_metric_cards(portfolios, name)
    render_divider(portfolios, name)

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        with st.container(border=True, height=670):
            st.markdown("#### Cumulative Portfolio Value")
            render_cumulative_returns(portfolios, name)
    with mcol2:
        with st.container(border=True, height=670):
            st.markdown("#### Percent Contribution of Risk")
            render_PCR(portfolios, name)

    render_visualize_metrics()