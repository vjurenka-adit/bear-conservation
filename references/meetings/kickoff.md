# Kickoff Meeting

2 Challenges in one. Sponsored by ARM.

[Link to the kickoff](https://drive.google.com/file/d/17IzvXvvwGgPfgJ0OI87hYldbdlKz8i2I/view)

## Bear ID project

### Stakeholders

- __Ed Miller__: Director and Software Developer
- __Melanie Clapham__: Director and Conservation Scientist

Small crew of 3 directors, started in 2018. Common solution to
automatically identify individual bears.
Develop non invasive to recognize, monitor and track bears.

2020: Developped ID models and iterated overtime.

### Why

Three ideas:

- Population monitoring
- indigenous conservation projects
- human bear conflicts (how to manage problem bears)

### Resources 

- [Research Paper - BearID](https://onlinelibrary.wiley.com/doi/full/10.1002/ece3.6840)

## Hack The Planet

[Website](https://www.hack-the-planet.io/)

How to use technology for good? Social projects as well. Projects in the
world of conservation too (poaching, human-wildlife conflict). How can
we help NGOs leverage technology to make their work easier?
Detecting poachers.
Camera trap with AI running on them (elephant, rhino, etc) to provide
real time insight. Scare light and sound to scare elephants from a
distance.

Romania bear-human conflicts.

Camera to analyze a video feed to run on hardware to trigger repellant responses.
Inflate a wacky wavy inflatable tube man as soon as we detect a bear
nearby.

- __Thijs Suijten__: Hacker
- __Tim van Deursen__: Hacker

[Camera used for the project](https://openmv.io/products/openmv-cam-rt?variant=39973964775518)

Night vision and day vision.
Infrared light dataset.

## Subteams

- __Classification on chip__ - TinyML - Is there an operating system?
  Smallest models possible that can run on non-linux micro controllers.
Use Edge Impulse to deploy models - Model pruning and quantization.
How well does it run on edge?
- __Face detection and segmentation__: Detect and segment the bear faces to
pass on the feature matching team.
Use SAM or YOLO to perform the segmentation. Use alignment if needed.
- __Identification - feature matching__: Try out various feature
extraction algorithms and matching techniques.
- __Bear detection on NPU__: Train TinyML models for efficient to run
the detection and identification pipeline on edge. NXP IMX93.
Start with the Turtles pipeline to see how things work.

## Questions

- How does the current Bear Identification models work?
- Identification models can be much slower than the classification models?

## Data

4 available datasets:

- BearID - identification
- HackThePlanet - classification
- Sensing Clues - classification
- SLU Multi species - classification (Swedish University of Agriculture
  - not that many images of bears)

Only still images available from BearID.

How do we test with a video feed?
HackThePlanet provides video frames too (exported as images).

## Homework - TODO

### By Tomorrow

- [Vote for a subteam and a role](https://docs.google.com/spreadsheets/d/193KSr5-qKMl_7HnegDY-t5ImH5vtJ41_0i0Acjv9WTA/edit#gid=0)
- Sign Participation agreement
- Vote for a weekly time in Slack

### By the weekend

- Assign Roles
- Schedule subteam meeting

### By next weekly meeting

- Explore the data / make exploratory notebooks
- Do research on what methods have been tried for this topic
- Fill the backlog on Notion
- Present the state of the art
- Present initial implementation plan/next steps
