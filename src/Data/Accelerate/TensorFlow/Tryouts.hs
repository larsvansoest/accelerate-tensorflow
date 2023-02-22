{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Data.Accelerate.TensorFlow.Tryouts where

import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.Representation.Shape (shapeType, ShapeR)
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Trafo.Desugar
import Data.Array.Accelerate.Trafo.Exp.Substitution
import Data.Accelerate.TensorFlow.Operation

mapXTimesTwoPlusOne :: forall env sh. Arg env (In sh Int64) 
  -> Arg env (Out sh Int64) -> OperationAcc TensorOp env ()
mapXTimesTwoPlusOne (ArgArray _ arrayR@(ArrayR sh ty) gvIn gvbIn) argOut
  | DeclareVars lhs  w  k  <- declareVars $ buffersR ty -- variable to new array
  , DeclareVars lhs' w' k' <- declareVars $ buffersR ty -- variable to new array
  = let
    nInt64 :: NumType Int64
    nInt64 = IntegralNumType TypeInt64
    sInt64 :: ScalarType Int64
    sInt64 = SingleScalarType (NumSingleType nInt64)
    arrayR' :: ArrayR (Array sh (Int64, Int64))
    arrayR' = ArrayR sh $ TupRpair (TupRsingle sInt64) (TupRsingle sInt64)
  in 
     -- Allocate new array
     aletUnique lhs (desugarAlloc arrayR (fromGrounds gvIn)) $
     Alet (LeftHandSideWildcard TupRunit) TupRunit
       (Exec -- Fill new new array with the number 2
         (TConstant sInt64 2) 
         (ArgArray 
           Out 
           (ArrayR sh (TupRsingle sInt64)) 
           (weakenVars w gvIn) -- Weaken variables with the new array
           (weakenVars w gvbIn) :>: ArgsNil
         )
       ) $
       Alet (LeftHandSideWildcard TupRunit) TupRunit
         (Exec -- (*2) Multiply input array with new array
           (TPrimFun (PrimMul nInt64))
           (ArgArray 
             In 
             arrayR' 
             (weakenVars w gvIn)
             (TupRpair (weakenVars w gvbIn) (k weakenId)) :>: weaken w argOut :>: ArgsNil
           )
         ) $
         -- Allocate new array
         aletUnique lhs' (desugarAlloc arrayR (fromGrounds (weakenVars w gvIn))) $
           Alet (LeftHandSideWildcard TupRunit) TupRunit
             (Exec -- Fill new array with the number 1
               (TConstant sInt64 1) 
               (ArgArray 
                 Out 
                 (ArrayR sh (TupRsingle sInt64)) 
                 (weakenVars (w' .> w) gvIn) -- Weaken variables with both new arrays
                 (weakenVars (w' .> w) gvbIn) :>: ArgsNil
               )
             ) $
            Exec -- (+1) Add new array to the result array of (*2)
              (TPrimFun (PrimAdd nInt64)) 
              (ArgArray 
                In 
                arrayR' 
                (weakenVars (w' .> w) gvIn) 
                (TupRpair (weakenVars (w' .> w) gvbIn) (k' weakenId)) 
                  :>: weaken (w' .> w) argOut :>: ArgsNil
              )

--mapXTimesTwoPlusOne :: forall env sh. Arg env (In sh Int64) -> Arg env (Out sh Int64) -> OperationAcc --TensorOp env ()
--mapXTimesTwoPlusOne (ArgArray _ arrayR@(ArrayR sh _) gvIn gvbIn) argOut = let
  -- nInt64 :: NumType Int64
  -- nInt64 = IntegralNumType TypeInt64

  -- sInt64 :: ScalarType Int64
  -- sInt64 = SingleScalarType (NumSingleType nInt64)

  -- bInt64 :: GroundR (Buffer Int64)
  -- bInt64 = GroundRbuffer sInt64

  -- arrayR' :: ArrayR (Array sh (Int64, Int64))
  -- arrayR' = ArrayR sh $ TupRpair (TupRsingle sInt64) (TupRsingle sInt64)

  -- gvIn' :: TupR (Var GroundR (env, Buffer Int64)) sh
  -- gvIn' = mapTupR (weaken (weakenSucc weakenId)) gvIn

  -- gvIn'' :: TupR (Var GroundR ((env, Buffer Int64), Buffer Int64)) sh
  -- gvIn'' = mapTupR (weaken (weakenSucc weakenId)) gvIn'

  -- varToI0 :: forall env. TupR (Var GroundR (env, Buffer Int64)) (Buffer Int64)
  -- varToI0 = TupRsingle $ Var bInt64 ZeroIdx

  -- gvbIn' :: TupR (Var GroundR (env, Buffer Int64)) (Buffer Int64, Buffer Int64)
  -- gvbIn' = TupRpair (mapTupR (weaken (weakenSucc weakenId)) gvbIn) varToI0

  -- gvbIn'' :: TupR (Var GroundR ((env, Buffer Int64), Buffer Int64)) (Buffer Int64, Buffer Int64)
  -- gvbIn'' = TupRpair (mapTupR (weaken (weakenSucc weakenId)) (mapTupR (weaken (weakenSucc weakenId)) gvbIn)) varToI0

  -- argOut' :: Arg (env, Buffer Int64) (Out sh Int64)
  -- argOut' = weaken (weakenSucc weakenId) argOut
  -- in -- eerst nieuwe buffer aanmaken, eerst array aanmaken van zelfde grootte
  --   Alet -- kan je gebruiken voor nieuwe variabelen of side effects uitvoeren en dan doorgaan met iets anders
  --   (LeftHandSideSingle bInt64) -- variable introduceren
  --   (TupRsingle Unique) -- uniqueness van nieuwe variabele
  --   (Alloc sh sInt64 (groundToExpVar (shapeType sh) gvIn))
  --   -- array vullen met tweeën
  --   $ Alet (LeftHandSideWildcard TupRunit)
  --     TupRunit
  --     (Exec (TConst sInt64 2) (ArgArray Out arrayR gvIn' varToI0 :>: ArgsNil)) -- (TupRsingle $ Var int64Buffer ZeroIdx) refereert naar een array
  --     -- keer twee
  --     $ Alet (LeftHandSideWildcard TupRunit)
  --       TupRunit
  --       (Exec
  --         (TPrimFun (PrimMul nInt64))
  --         (ArgArray In arrayR' gvIn' gvbIn' :>: argOut' :>: ArgsNil)
  --       )
  --       -- nieuwe array aanmaken van zelfde grootte
  --       $ Alet
  --           (LeftHandSideSingle bInt64)
  --           (TupRsingle Unique)
  --           (Alloc sh sInt64 (groundToExpVar (shapeType sh) gvIn'))
  --           -- array vullen met 1'en
  --           $ Alet (LeftHandSideWildcard TupRunit)
  --             TupRunit
  --             ( Exec
  --               (TConst sInt64 1)
  --               (ArgArray Out arrayR gvIn'' varToI0 :>: ArgsNil)
  --             )
  --             -- plus 1
  --             $ Exec
  --                 (TPrimFun (PrimAdd nInt64))
  --                 (ArgArray In arrayR' gvIn'' gvbIn''
  --                   :>: weaken (weakenSucc weakenId) argOut'
  --                   :>: ArgsNil
  --                 )
