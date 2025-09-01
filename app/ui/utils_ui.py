import streamlit as st


def render_info_box(message:str, icon:str=None, height:int=300, margin_top:str="0px", max_width:str="900px"):
    """
    Draw a centered info-like box, vertically centered inside a flex container of given height.

    Parameters: 
    - message: The message to be displayed inside the info box
    - icon: Optional icon to be displayed alongside the message
    - height: px height of the placeholder container
    - margin_top: CSS margin-top value for pushing the whole placeholder down (e.g. "20px")
    - max_width: max width of the info box (keeps it from being full width)
    """
    bg = "#2171b5"   
    txt = "#ffffff"
    border = '#4292c6'

    if icon:
        content = f'<span style="font-size: 28px; margin-right: 8px;">{icon}</span>{message}'
    else:
        content = message

    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; align-items:center; height:{height}px; margin-top:{margin_top};">
          <div style="
              background-color: {bg};
              color: {txt};
              padding: 16px 22px;
              border-radius: 10px;
              border: 1px solid {border};
              font-size: 16px;
              max-width: {max_width};
              text-align: center;
              box-shadow: rgba(0,0,0,0.08) 0px 2px 6px;
              pointer-events: none;
          ">
            {content}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )