This project extracts insights from Google reviews about McDonald's in Selangor.

- Firstly, it's extracting the McDonald's list in Selangor.
- The reason for doing so is to find out the worst McDonald's service in Selangor. (Rating < 2.5)
- The extract insight output will be saved in McD Selangor List.xlsx.
- The cleaning and standardization process by using Script _Data Exploration for McD Selangor List.xlsx save in Cleaned_McDonald_Selangor_List.xlsx.
- Using Cleaned_McDonald_Selangor_List.xlsx, we can find out the worst McDonald's service in Selangor. (Rating < 2.5)

- Second, we know we have 8 McDonald's branches with the worst ratings.
- By taking customer reviews from there, we created a dataset for each branch.
( McDonald AEON BiG Bukit Rimau DT, McDonald BHP Jalan Genting Kelang DT Store 476, McDonald Desa Petaling SF, McDonald Puchong Perdana SF, McDonald Shell Saujana Utama DT, McDonald Shell Taman Desa DT, McDonald The Mines, McDonald’s Petronas Kota Damansara DT )
- Clean up and standardize the process of using Script_Data Collection to store the dataset in McDonalds_Reviews_Details.xlsx.

- Third, we want to pre-process the text because the dataset is ready (McDonalds_Reviews_Details.xlsx).
- Using Script_Text Processing, we create pre-processed text that is stored in Processed_McDonalds.xlsx.