# AtlasA2 Hardware Basics

## Programming Model

<https://www.hiascend.com/en/document/redirect/CannCommunityAscendCOnHost>

## Architecture Diagram

<https://www.hiascend.com/en/document/redirect/CannCommunityAscendCOnHost>

## Memory Units

| Level     | Subdivision | Space                            | Description                                                                                                                       |
| --------- | ----------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| GM        |             |                                  |                                                                                                                                   |
| L2 Cache  |             |                                  |                                                                                                                                   |
| L1 Buffer |             | 512KB                            | Temporarily stores data reused repeatedly inside AI Core; minimum access granularity is cacheline-32B; usually split into 4 x 128KB, L1A pingpong, L1B pingpong |
| L0 Buffer | L0A/L0B     | 64KB each<br />usually split into two 32KB each | Input of Cube<br />usually split into 4 x 32KB, L0A pingpong, L0B pingpong                                                |
|           | L0C         | 128KB                            | Cube output,<br />double buffer can be set explicitly; usually enabled via unit-flag for compute-while-move                    |
|           | BT          | 1KB                              | Stores bias                                                                                                                       |
|           | FB          | 7KB                              | Stores quantization parameters, relu parameters                                                                                   |
| UB        |             | 192KB<br />usually split into two 96KB | Vector storage                                                                                                               |

| Logical Position | Physical Position | Load Instruction                     | Supported Format                     |
| ---------------- | ----------------- | ------------------------------------ | ------------------------------------ |
| GM               | GM                | DataCopy                             | RowMajor/ColumnMajor/nZ/zN            |
| A1/B1/C1         | L1                | DataCopy                             | zN(RowMajor)/nZ(ColumnMajor)/ND(m=1) |
| A2               | L0A               | LoadData                             | zZ                                   |
| B2               | L0B               | LoadData                             | nZ                                   |
| CO1              | L0C               | -                                    | zN                                   |
| CO2              | BiasTable Buffer  | DataCopy                             | -                                    |
| C2PIPE2GM        | Fixpipe Buffer    | DataCopy                             | -                                    |
| tbufVECIN        | UB                | GM->UB DataCopy<br />UB->UB DataCopy | ND/NZ                                |
| tbufVECOUT       | UB                |                                      | ND/NZ                                |
| tbufVECCALC      | UB                |                                      | ND/NZ                                |
|                  |                   |                                      |                                      |
