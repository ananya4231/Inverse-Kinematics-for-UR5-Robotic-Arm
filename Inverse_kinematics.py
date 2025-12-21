import modern_robotics as mr
import numpy as np

'''Compute final joint angles for the UR5 robot by entering an initial guess (in degrees) when prompted, an input sequence will look like:
    This is an example
    Enter joint angle 1 : 130
    Enter joint angle 2 : 70
    Enter joint angle 3 : -140
    Enter joint angle 4 : -110
    Enter joint angle 5 : 40
    Enter joint angle 6 : 100
    
    This combination takes 4 iterations to converge + initial guess
    
    Output in radians'''

def IKinBodyIterates(theta_list):
    '''This is the desired orientation of the joint angles, given in the question'''
    T_sd = np.array([
        [0, 1, 0, -0.5],
        [0, 0, -1, 0.1],
        [-1, 0, 0, 0.1],
        [0, 0, 0, 1]
    ])

    '''This is the end-effector configuration when all the joint angles are 0'''
    M_matrix = np.array([
        [-1, 0, 0, 0.817],
        [0, 0, 1, 0.191],
        [0, 1, 0, -0.005],
        [0, 0, 0, 1]
    ])

    '''Screw axes in the base frame as given in the question'''
    screw_axes_b = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, -1, 0],
        [0, 1, 1, 1, 0, 1],
        [0.191, 0.095, 0.095, 0.095, -0.082, 0],
        [0, -0.817, -0.392, 0, 0, 0],
        [0.817, 0, 0, 0, 0, 0]
    ])

    '''Allowed error in for omega (angular velocity)'''
    e_w = 0.001
    '''Allowed error in for v (linear velocity)'''
    e_v = 0.0001

    '''Calculating T_sb, and then T_bs which is the same as T_sb inverse, as required.
        mr.FKinBody(M_matrix, screw_axes_b, theta_list)) will calculate the homogeneous transformation T_sb
        mr.TransInv() will then calculate the inverse of the homogeneous transformation'''

    T_sb = mr.FKinBody(M_matrix, screw_axes_b, theta_list)
    T_bs = mr.TransInv(T_sb)  # Finding the inverse here

    '''Finding T_bd which helps us figure out how far we are from the desired orientation
        by multiplying T_bs and T_sd'''
    T_bd = np.matmul(T_bs, T_sd)

    '''Using the matrix logarithm to first get the so(3) representation'''
    Vb = mr.MatrixLog6(T_bd)

    '''Getting Vb back into a 6x1 vector'''
    Vb_vector = mr.se3ToVec(Vb)

    '''err is a boolean value that checks if the error is still big enough by calculating the magnitude
       of the angular and linear velocities from the twist Vb'''
    err = (np.linalg.norm([Vb_vector[0], Vb_vector[1], Vb_vector[2]]) > e_w) or (np.linalg.norm([Vb_vector[3], Vb_vector[4], Vb_vector[5]]) > e_v)

    '''Counter variable that will terminate the loop after 50 iterations if solution has not converged'''
    count = 0

    '''Matrix that holds all the iterations of the joint angles'''
    joint_angles_matrix = np.empty((0, 6))

    '''Loop that carries out the iterations'''
    while err == True and count < 1000:
        '''Prints the iteration number'''
        print("Iteration: ", count)
        '''This is the pseudoinverse of the Jacobian matrix
            mr.JacobianBody(screw_axes_b,theta_list) will find the Jacobian, and this is used to find the pseudoinverse'''
        pseudoinverse_J = np.linalg.pinv(mr.JacobianBody(screw_axes_b, theta_list))

        '''Carrying out the Newton Raphson process here
            the next set of joint angles is calculated by adding the old set of angles to the product of
            the pseudoinverse and the twist vector Vb_vector'''

        theta_list = theta_list + np.matmul(pseudoinverse_J, Vb_vector)  # prints joint angles
        theta_list = (theta_list + np.pi) % (2 * np.pi) - np.pi

        '''Adding the new row of joint angles to the matrix keeping track of all joint values'''
        joint_angles_matrix = np.round_(np.vstack((joint_angles_matrix, theta_list)),3)

        print("joint vector: ",np.round_(theta_list, 3))  # printing theta list with all values rounded to 3 decimal places

        '''Updating values using the new joint angles for the next iteration'''
        T_sb = mr.FKinBody(M_matrix, screw_axes_b, theta_list)
        print("SE(3) end−effector config:")
        print(np.round_(T_sb, 3))  # Rounding value to 3 decimal spaces

        T_bs = mr.TransInv(T_sb)  # Calculating the inverse of T_sb here
        T_bd = np.matmul(T_bs, T_sd)

        '''Resetting the error based on the twist Vb_vector'''
        print("angular error magnitude ∣∣omega_b∣∣:", round(np.linalg.norm([Vb_vector[0], Vb_vector[1], Vb_vector[2]]),5))
        print("linear error magnitude ∣∣v_b∣∣:", round(np.linalg.norm([Vb_vector[3], Vb_vector[4], Vb_vector[5]]), 5))
        err = (np.linalg.norm([Vb_vector[0], Vb_vector[1], Vb_vector[2]]) > e_w) or (np.linalg.norm([Vb_vector[3], Vb_vector[4], Vb_vector[5]]) > e_v)

        '''Finding the new twist'''
        Vb = mr.MatrixLog6(T_bd)  # so(3) representation
        Vb_vector = mr.se3ToVec(Vb)  # Getting Vb back into a 6x1 vector
        print("error twist: ", np.round_(Vb_vector, 5))

        print()
        count = count + 1  # incrementing counter
    # Printing Joint angles
    print("All iterations of joint angles:")

    print(np.round_(joint_angles_matrix, 3))
    file = open("iterates.csv", "w")
    for i in range(0,count):
        for j in range(0,6):
            if(j!=5):
                file.write(str(joint_angles_matrix[i, j]) + ",")
            else:
                file.write(str(joint_angles_matrix[i, j]))
        file.write("\n")
    file.close()

angles = []
for i in range (0,6):
    joint_angle = float(input("Enter joint angle "+ str(i+1) + " : "))
    angles.append(np.deg2rad(joint_angle))
#print(angles)

theta_list = np.array(angles)
IKinBodyIterates(theta_list)



