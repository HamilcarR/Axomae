#include "Test.h"
#include "../includes/BoundingBox.h"

#define m_rand rand()%100 + 1 


constexpr unsigned ITERATION_NUMBER = 5; 
std::vector<float> vertices = {
		-1 , -1 , -1 	,  // 0
		1 , -1 , -1 	,  // 1
		-1 , 1 , -1 	,   // 2
		1 , 1 , -1 		,   // 3
		-1 , -1 , 1 	,   // 4
		1 , -1 , 1 		,   // 5
		-1 , 1 , 1 		,    // 6
		1 , 1 , 1 		    //7
};  






TEST(BoundingBoxTest , productOperatorTest){
    BoundingBox B1(vertices); 
    srand(time(nullptr)); 
    std::vector<std::pair<glm::mat4 , glm::vec3[3]>> matrix_result ; //first = matrices , second = result matrix x coords => {min_coords , max_coords , center}   
    matrix_result.resize(ITERATION_NUMBER); 
    for(unsigned i = 0 ; i < ITERATION_NUMBER; i++){
        glm::vec4 row1(m_rand , m_rand , m_rand , m_rand);
        glm::vec4 row2(m_rand , m_rand , m_rand , m_rand);
        glm::vec4 row3(m_rand , m_rand , m_rand , m_rand);
        glm::vec4 row4(m_rand , m_rand , m_rand , m_rand); 
        glm::mat4 mat(row1,row2,row3,row4);
        glm::vec3 min_coords = mat * glm::vec4(B1.getMinCoords() , 1.f); 
        glm::vec3 max_coords = mat * glm::vec4(B1.getMaxCoords() , 1.f); 
        glm::vec3 center = mat * glm::vec4(B1.getPosition() , 1.f);
        matrix_result[i].first = mat ; 
        matrix_result[i].second[0] = min_coords; 
        matrix_result[i].second[1] = max_coords; 
        matrix_result[i].second[2] = center;  
    }
    for(unsigned i = 0 ; i < ITERATION_NUMBER ; i++){
        BoundingBox B2 = matrix_result[i].first * B1 ; 
        glm::vec3 min_coords_B1 = matrix_result[i].second[0];
        glm::vec3 max_coords_B1 = matrix_result[i].second[1]; 
        glm::vec3 center_B1 = matrix_result[i].second[2];
        EXPECT_EQ(B2.getMinCoords() , min_coords_B1); 
        EXPECT_EQ(B2.getMaxCoords() , max_coords_B1); 
        EXPECT_EQ(B2.getPosition() , center_B1); 
    }


}

