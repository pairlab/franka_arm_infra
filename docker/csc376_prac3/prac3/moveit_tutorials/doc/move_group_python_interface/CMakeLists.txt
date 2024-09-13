catkin_install_python(PROGRAMS
#  scripts/move_group_python_interface_tutorial.py
#  scripts/pick_n_place.py
#  scripts/csc376_backend.py
#  scripts/csc376_practical3.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
