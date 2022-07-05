import pickle
import numpy as np
import sklearn
import streamlit as st

pickle_in = open(r'C:\Users\sahil\Documents\360 DigitMG\Project\project attempt2\edtech_flask.pkl', 'rb')
final_model = pickle.load(pickle_in)


def main():
    
    st.title('Edtech Course Price')
    html_temp = """
    <div style="background-color:Blue;padding:10px">
    <h2 style="color:white;text-align:center;">Edtech App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    institute_brand_value = st.selectbox("institute_brand_value", ['High', 'Low'])
    if(institute_brand_value =='High'):
        institute_brand_value = 1
    else:
        institute_brand_value = 0
    
    instructors_grade = st.selectbox("instructors_grade", ['A', 'B'])
    if(instructors_grade =='A'):
        instructors_grade = 1
    else:
        instructors_grade = 0
        
    number_of_instructors = st.slider('number_of_instructors',1,2,3)
    
    course_Level = st.selectbox("course_Level", ['Advance', 'Intermediate','Beginners'])
    if(course_Level =='Advance'):
        course_Level = 2
    elif(course_Level =='Intermediate'):
        course_Level = 1
    else:
        course_Level = 0
        
    course_length = st.slider("course_length",20,120,240)    
        
    cost_of_course_curricullum = st.selectbox("cost_of_course_curricullum", ['High', 'Medium','Low'])
    if(cost_of_course_curricullum =='High'):
        cost_of_course_curricullum = 2
    elif(cost_of_course_curricullum =='Medium'):
        cost_of_course_curricullum = 1
    else:
         cost_of_course_curricullum = 0
    
    study_material_cost = st.text_input("study_material_cost")
    
    Office_rent = st.slider("Office_rent",15000,50000,100000)
    
    Infrastructure_cost = st.selectbox("Infrastructure_cost", ['High', 'Low'])
    if(Infrastructure_cost =='High'):
        Infrastructure_cost = 1
    else:
        Infrastructure_cost = 0
        
    Office_electricity_charges = st.text_input("Office_electricity_charges")
    
    Misclleneous_expense = st.text_input("Misclleneous_expense")
    
    cost_of_acquisition = st.text_input("cost_of_acquisition")
    
    competition_level = st.selectbox("competition_level", ['High', 'Medium','Low'])
    if(competition_level =='High'):
       competition_level = 2
    elif(competition_level =='Medium'):
        competition_level = 1
    else:
        competition_level = 0
    
    
    course_title_Data_Science = st.selectbox("course_title", ['Data_Science', 'Tableau','Artificial_Intelligence'])
    if(course_title_Data_Science =='Data_Science'):
       course_title_Data_Science = 1
       course_title_Tableau = 0
    elif(course_title_Data_Science =='Tableau'):
        course_title_Data_Science = 0
        course_title_Tableau = 1
    else:
        course_title_Data_Science = 0
        course_title_Tableau = 0 
        
    course_market_School = st.selectbox("course_market", ['School', 'student','College'])
    if(course_market_School =='School'):
       course_market_School = 1
       course_market_student = 0
    elif(course_market_School =='student'):
        course_market_School = 0
        course_market_student = 1
    else:
        course_market_School = 0
        course_market_student = 0 
        
    online_live_class_Yes = st.selectbox("online_live_class", ['Yes', 'No'])
    if(online_live_class_Yes =='Yes'):
        online_live_class_Yes = 1
    else:
        online_live_class_Yes = 0
    
    
    online_pre_recorded_sessions_Yes = st.selectbox("online_pre_recorded_sessions" , ['Yes', 'No'] )
    if(online_pre_recorded_sessions_Yes =='Yes'):
        online_pre_recorded_sessions_Yes = 1
    else:
        online_pre_recorded_sessions_Yes = 0    
        
        
    Offiline_classes_Yes = st.selectbox("Offiline_classes" , ['Yes', 'No'] )
    if(Offiline_classes_Yes =='Yes'):
        Offiline_classes_Yes = 1
    else:
        Offiline_classes_Yes = 0    
   
    state_Hyderabad = st.selectbox("state", ['Hyderabad', 'Delhi'])
    if(state_Hyderabad =='Hyderabad'):
        state_Hyderabad = 1
    else:
        state_Hyderabad = 0    

    support_Staff_required_Yes = st.selectbox("support_Staff_required", ['Yes', 'No'])
    if(support_Staff_required_Yes =='Yes'):
        support_Staff_required_Yes = 1
    else:
        support_Staff_required_Yes = 0
    
    
    certification_Yes = st.selectbox("certification" , ['Yes', 'No'] )
    if(certification_Yes =='Yes'):
        certification_Yes = 1
    else:
        certification_Yes = 0    
        
        
    Placement_Yes = st.selectbox("Placement" , ['Yes', 'No'] )
    if(Placement_Yes =='Yes'):
        Placement_Yes = 1
    else:
        Placement_Yes = 0 
    

    result=""
    if st.button("Predict"):
        result= final_model.predict([[institute_brand_value, instructors_grade, number_of_instructors,
                course_Level, course_length, cost_of_course_curricullum,
                study_material_cost, Office_rent, Infrastructure_cost,
                Office_electricity_charges, Misclleneous_expense,
                cost_of_acquisition, competition_level, course_title_Data_Science,
                course_title_Tableau, course_market_School, course_market_student,
                online_live_class_Yes, online_pre_recorded_sessions_Yes,
                Offiline_classes_Yes, state_Hyderabad, support_Staff_required_Yes,
                certification_Yes, Placement_Yes]])
        output =round(result[0])
        st.success('The product price is {}'.format(output))





if __name__=="__main__":
    main()