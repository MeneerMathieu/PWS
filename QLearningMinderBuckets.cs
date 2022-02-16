using System.Collections;
using System.IO;
using UnityEngine;

public class QLearningMinderBuckets : MonoBehaviour
{
    //Het maken van input fields in unity.
    [SerializeField] private Transform Car;
    [SerializeField] private float dfactor;
    [SerializeField] private Rigidbody Carbody;
    [SerializeField] private float gamma;
    [SerializeField] private int updateEvery = 100;
    [SerializeField] private float epsilon = 1;
    [SerializeField] private float learningRate = (float)0.1;
    [SerializeField] private float discount = (float)0.95;

    [SerializeField] private int runs = 10000;
    [SerializeField] private int startEpsilonDecaying = 1;
    [SerializeField] private float endEpsilonDecaying = 0.5f;
    [SerializeField] private float minimalEpsilon = 0.01f;
    
    //Settings voor het epsilon greedy algoritme.
    int EndEpsilonDecaying;
    float epsilonDecayValue;
    bool died;
    static int timeSinceDied = 0;


    private float[] currentState = new float[4];
    int[] discreteState = { 0, 0, 0, 0 };
    //angle, distance, angular velocity
    float[,,,,] q_matrix;
    int count;
    int currentRunCount = 0;
    int resets = 0;
    int action = 0;
    static bool saved = false;

    ArrayList recentRuns = new ArrayList();

    int[] newDiscreteState = { 0, 0, 0, 0 };

    void Start()
    {
        //het maken van een nieuwe 5 dimensionale tabel. De cijfers representeren de lengte in een bepaalde richting.
        q_matrix = new float[2, 6, 3, 6, 3];
        //action, angle, distance, angular velocity, velocity


        //Het laden van de originele staat van de auto.
        newDiscreteState[0] = Angle(state(Car, Carbody));
        newDiscreteState[1] = Distance(state(Car, Carbody));
        newDiscreteState[2] = aVel(state(Car, Carbody));
        newDiscreteState[3] = vel(state(Car, Carbody));
        discreteState = newDiscreteState;
        EndEpsilonDecaying = (int)(runs * endEpsilonDecaying);
        epsilonDecayValue = epsilon / (EndEpsilonDecaying - startEpsilonDecaying);
    }

    // Wordt elk frame opgeroepen door Unity
    void FixedUpdate()
    {
        run();


    }

    private void run()
    {
        //het laden van info uit unity voor een nieuwe discrete state.
        timeSinceDied++;
        float[] newstate = state(Car, Carbody);
        newDiscreteState[0] = Angle(newstate);
        newDiscreteState[1] = Distance(newstate);
        newDiscreteState[2] = aVel(newstate);
        newDiscreteState[3] = vel(newstate);
        float reward = getReward(newDiscreteState);

        if (died)
        {
            discreteState = newDiscreteState; //om te zorgen dat dingen niet crashen.
            reward = -400; //straf voor doodgaan.
            died = false;
        }

        float newQ;
        float maxFutureQ = maxQ(q_matrix, newDiscreteState);
        float currentQ = getQ(q_matrix, discreteState, action);

        newQ = currentQ + learningRate * (reward + discount * maxFutureQ - currentQ); //Het berekenen van de nieuwe Q waarde op een bepaalde plek.
        q_matrix[action, discreteState[0], discreteState[1], discreteState[2], discreteState[3]] = newQ; //Het invullen van de q-matrix
        
            

        
        Debug.Log("Q: " + newQ  + " Reward:  " + reward + " Iteration: " + count + "  Resets: " + resets);

        if (checkDied(newstate))
        {
            //Alles wat er moet worden gedaan/bijgehouden als de auto "doodgaat"
            timeSinceDied = 0;
            died = true;
            resets++;
            count++;
            recentRuns.Add(currentRunCount);
            if (resets % 10 == 0 && !saved)
            {
                saved = true;
                saveRuns(recentRuns);
                foreach (int i in recentRuns)
                {
                    Debug.Log(i);
                }
                recentRuns.Clear();
                
            }else if(!(resets % 10 == 0))
            {
                saved = false;
            }
            currentRunCount = 0;
            return;
        }
        //Het updaten van de discrete state. Je gaat 1 stapje vooruit in de tijd.
        discreteState = newDiscreteState;
        count++;

        if (UnityEngine.Random.Range(0f, 1f) > epsilon)
        {
            action = maxAction(q_matrix, discreteState);
            Debug.Log("Action chosen: " + action);
        }
        else
        {
            action = UnityEngine.Random.Range(0, 2);
        }

        HandleMotor(action);
        currentRunCount++;





    }

    string path = @"C:\Users\Mathieu\Programmeren\test\test.csv";

    private void saveRuns(ArrayList recentRuns)
    //het opslaan van data in een csv file, zodat we makkelijk grafiekjes kunnen maken in spreadsheets.
    {
        int sum = 0;
        foreach(int count in recentRuns)
        {
            sum = sum + count;
        }
        float avg = sum / recentRuns.Count;
        if (!File.Exists(path))
        {
            File.Create(path);
        }
        string avgstring = avg.ToString();
        using(StreamWriter sw = File.AppendText(path))
        {
            sw.WriteLine(avgstring);
            Debug.Log("written to file");
        }
    }

    private float getReward(int[] newDiscreteState)
    {
        float state = newDiscreteState[0];
        float reward = 2.5f - Mathf.Cos(state - 2.5f);
        return reward;
        //de reward is 1 voor elke keer dat de auto niet "doodgaat"
        
    }

    private bool checkDied(float[] state)
    {
        //het checken of de auto is doodgegaan in de laatste stap. state[0] is de hoek, state[1] is de afstand tot het midden.
        if (state[0] < -13f || state[0] > 13f || state[1] < -1.5f || state[1] > 1.5f)
        {
            resetPosition(Car);
            return true;
        }
        return false;
    }

    private void resetPosition(Transform agent)
    {
        //het reseten van de plaats van de auto in unity. De auto krijgt bij elke reset een random hoek om ervoor te zorgen dat hij niet
        //elke keer in dezelfde situatie is.
        Debug.Log("reset");
        Quaternion resetRotation = Quaternion.Euler(0, 0, 0);
        agent.position = new Vector3(0, 0.41f, 0);
        agent.rotation = resetRotation;
        Carbody.velocity = new Vector3(0, 0, 0);
        Carbody.angularVelocity = new Vector3(0, 0, 0);
        float rand;
        rand = UnityEngine.Random.Range(-6f, 6f);
        agent.Rotate(rand, 0f, 0f, Space.World);

        if (EndEpsilonDecaying >= resets && resets >= startEpsilonDecaying && epsilon > minimalEpsilon)
        {
            epsilon = epsilon - epsilonDecayValue;
        }
    }

    private float getQ(float[,,,,] q_matrix, int[] discreteState, int action)
    {
        //het halen van een q-waarde uit de q-matrix.
        Debug.Log(action + " " + discreteState[0] + " " + discreteState[1] + " " + discreteState[2] + " " + discreteState[3]);
        return q_matrix[action, discreteState[0], discreteState[1], discreteState[2], discreteState[3]];
    }

    private int maxAction(float[,,,,] matrix, int[] discreteState)
    {
        //het verkrijgen van de actie met de hoogste q-waarde.
        float highest = matrix[0, discreteState[0], discreteState[1], discreteState[2], discreteState[3]];
        int highaction = 0;
        for (int i = 0; i < 2; i++)
        {
            Debug.Log(i + " " + matrix[i, discreteState[0], discreteState[1], discreteState[2], discreteState[3]]);
            if (matrix[i, discreteState[0], discreteState[1], discreteState[2], discreteState[3]] >= highest)
            {
                highest = matrix[i, discreteState[0], discreteState[1], discreteState[2], discreteState[3]];
                highaction = i;
            }
        }
        return highaction;
    }

    private float maxQ(float[,,,,] matrix, int[] discreteState)
    {
        //het verkrijgen van de hoogst mogelijke q-waarde.
        float highest = matrix[0, discreteState[0], discreteState[1], discreteState[2], discreteState[3]];
        for (int i = 0; i < 2; ++i)
        {
            if (matrix[i, discreteState[0], discreteState[1], discreteState[2], discreteState[3]] > highest)
            {
                highest = matrix[i, discreteState[0], discreteState[1], discreteState[2], discreteState[3]];
            }
        }
        return highest;
    }

    private float[] state(Transform car, Rigidbody carbody)
    {
        //het halen van informatie uit unity.
        float[] state = new float[4];
        state[0] = car.eulerAngles.x;
        state[1] = car.position.z;
        state[2] = carbody.angularVelocity.x;
        state[3] = carbody.velocity.z;


        if (state[0] > 180)
        {
            state[0] = -(360 - state[0]);
        }

        return state;
    }

    //De onderstaande paar functies zijn voor het maken van de "buckets"
    private int aVel(float[] state)
    {
        float aVel = state[2];
        if (aVel > 2.5 || aVel < -2.5)
        {
            aVel = (aVel / Mathf.Abs(aVel)) * 2.5f;
        }
        aVel = aVel + 2.5f;

        return Mathf.RoundToInt(aVel);
    }

    private int Angle(float[] state)
    {
        float angle = state[0];

        angle = angle * 0.125f;
        angle = angle + 2.5f;

        if (angle > 5)
        {
            angle = 5;
        }
        return Mathf.RoundToInt(angle);
    }

    private int Distance(float[] state)
    {
        float distance = state[1];
        distance = distance * 0.75f + 1.5f;
        return Mathf.RoundToInt(distance);
    }

    private int vel(float[] state)
    {
        float vel = state[3];
        vel = vel * 1.5f + 1.5f;
        if (vel > 2)
        {
            vel = 2;
        }
        if (vel < 0)
        {
            vel = 0;
        }
        return Mathf.RoundToInt(vel);
    }


    [SerializeField] private float motorForce;
    [SerializeField] private WheelCollider leftWheelCollider;
    [SerializeField] private WheelCollider rightWheelCollider;
    //Hiermee wordt de informatie weer terug gegeven aan unity
    private void HandleMotor(int action)
    {
        float input = 0;
        switch (action)
        {
            case 0:
                input = -1f;
                break;
            case 1:
                input = 1f;
                break;
            default:
                input = 0;
                break;
        }
        leftWheelCollider.motorTorque = input * motorForce;
        rightWheelCollider.motorTorque = input * motorForce;

    }
}
